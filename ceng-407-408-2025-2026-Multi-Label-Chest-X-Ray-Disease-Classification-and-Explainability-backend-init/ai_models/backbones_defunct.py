import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from ai_models.CBAM import cbam


class DenseNetCBAMBackbone(nn.Module):
    def __init__(self):
        super(DenseNetCBAMBackbone,self).__init__()

        # load pretrained densenet
        base = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1)



        # freeze first layers
        for name, module in base.features.named_children():
            if name in ["conv0", "norm0", "denseblock1", "transition1"]:
                for p in module.parameters():
                    p.requires_grad = False

        # extract features to find denseblocks
        features = list(base.features.children())


        # make a list for the new features
        CBAM_features = []

        # find all denseblocks and add placeholder tuple
        for module in features:
            CBAM_features.append(module)

            if module.__class__.__name__.lower().startswith("denseblock"):
                CBAM_features.append(("CBAM_PLACEHOLDER", None))

        # swap placeholders in feature list with Identity after denseblocks in actual model modules
        CBAM_modules = nn.ModuleList()
        for feature in CBAM_features:
            if isinstance(feature, tuple) and feature[0] == "CBAM_PLACEHOLDER":
                CBAM_modules.append(nn.Identity())
            else:
                CBAM_modules.append(feature)

        #unpack list and add(swap) the new modules in
        self.features = nn.Sequential(*CBAM_modules)
        self.swap_CBAM_modules_with_identity()

    def forward(self, x):
        output = self.features(x)
        return output


    def swap_CBAM_modules_with_identity(self):

        dummy_input = torch.zeros(1, 3, 224, 224)
        last_feature = dummy_input  # Tracks the output tensor of the previous module

        new_module_list = []
        with torch.no_grad():
            for feature in self.features:
                if isinstance(feature, nn.Identity):

                    # get the channel count from the last module output
                    channel = last_feature.shape[1]
                    new_CBAM_feature = cbam(in_channels=channel)

                    # add the new CBAM layer to the list
                    new_module_list.append(new_CBAM_feature)

                    # Update the tensor size by applying the new CBAM.
                    # This ensures the subsequent module (Transition Layer) receives the correct input size.
                    last_feature = new_CBAM_feature(last_feature)

                else:
                    # add the original module
                    new_module_list.append(feature)

                    # Apply the original module to the dummy tensor to track the size.
                    # This is done *for every non-Identity module* to maintain the correct last_feature size.
                    last_feature = feature(last_feature)

        self.features = nn.Sequential(*new_module_list)

