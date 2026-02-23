import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class VCA(nn.Module):
    def __init__(self, clip_model_name="ViT-L/14", is_predict_two=True, num_class=None, device="cpu"):
        super(VCA, self).__init__()
        self.device = device
        self.is_predict_two = is_predict_two
        self.num_class = num_class
        output_dim = 2 if self.is_predict_two else 1
        # Load the CLIP model dynamically based on the passed model name
        self.clip_model, _ = clip.load(clip_model_name, device=self.device, jit=False)
        self.clip_input_resolution = self.clip_model.visual.input_resolution  # 224 for ViT-L/14

        # Freeze the CLIP model's parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Define your additional layers for processing
        # Assuming the CLIP model output feature size, adjust if necessary
        if clip_model_name=="ViT-L/14":
            clip_output_dim = 768
        else:
            clip_output_dim = 512  # This is an example, adjust based on your CLIP model's output


        # Transformer layer configuration
        transformer_layer_config_LA = nn.TransformerEncoderLayer(
            d_model=clip_output_dim,  # Adjust based
            nhead=4,  # Number of attention heads
            dim_feedforward=512,  # Feedforward dimension
            dropout=0.1,  # Dropout rate
        )

        transformer_layer_config = nn.TransformerEncoderLayer(
            d_model=clip_output_dim,  # Adjust based
            nhead=1,  # Number of attention heads
            dim_feedforward=512,  # Feedforward dimension
            dropout=0.1,  # Dropout rate
        )

        # Replacing LA, BA, BL, CE with transformer blocks
        self.LA = nn.TransformerEncoder(transformer_layer_config_LA, num_layers=2)

        self.BA = nn.Sequential(nn.TransformerEncoder(transformer_layer_config, num_layers=1),
                                nn.Linear(clip_output_dim, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.1))
        self.BL = nn.Sequential(nn.TransformerEncoder(transformer_layer_config, num_layers=1),
                                nn.Linear(clip_output_dim, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.1))

        transformer_layer_config_CE = nn.TransformerEncoderLayer(
            d_model=clip_output_dim + 512,  # Adjust based the model's output
            nhead=2,  # Number of attention heads
            dim_feedforward=512,  # Feedforward dimension
            dropout=0.1,  # Dropout rate
        )
        self.CE = nn.Sequential(nn.TransformerEncoder(transformer_layer_config_CE, num_layers=2),
                                nn.Linear(clip_output_dim + 512, clip_output_dim + 512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5))

        self.LastFC = nn.Linear(clip_output_dim + 512, output_dim)  # Adjust the input dimension as necessary

    def forward(self, x, return_features=False):
        x = self.preprocess_for_clip(x)  # Preprocess input

        features = self.clip_model.encode_image(x)  # Extract features with CLIP
        # Ensure dtype consistency with downstream Transformer layers (which are float32 by default)
        features = features.float()

        y_high = features.view(features.size(0), -1)  # Flatten the features tensor

        # TransformerEncoder expects (seq_len, batch, embed_dim); treat each sample as seq_len=1
        y_high = y_high.unsqueeze(0)  # (1, batch, embed_dim)

        # Process through the parallel FC layers
        y_LA = self.LA(y_high)           # (1, batch, embed_dim)
        y_BA = self.BA(y_LA).squeeze(0)  # (batch, 256)
        y_BL = self.BL(y_LA).squeeze(0)  # (batch, 256)
        y_LA = y_LA.squeeze(0)           # (batch, embed_dim)

        # Concatenating outputs for CE input
        combined_features = torch.cat([y_LA, y_BA, y_BL], dim=1).unsqueeze(0)  # (1, batch, embed_dim+512)
        y_CE = self.CE(combined_features).squeeze(0)  # (batch, embed_dim+512)

        regression = self.LastFC(y_CE)

        if return_features:
            features_dict = {
                'LA': y_LA,
                'BL': y_BL,
                'BA': y_BA,
                'CE': y_CE,  # Assuming you want the output before the final FC layer
            }
            return regression, features_dict

        # Handling the prediction logic based on is_predict_two and num_class
        if self.is_predict_two and self.num_class is None:
            output1, output2 = regression[:, 0], regression[:, 1]
            return output1, output2
        else:
            return regression

    def preprocess_for_clip(self, x):
        # Ensure input images match the resolution CLIP expects
        res = self.clip_input_resolution
        if x.shape[-2] != res or x.shape[-1] != res:
            x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
        return x
