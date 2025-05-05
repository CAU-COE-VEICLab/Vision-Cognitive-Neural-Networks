# --------------------------------------------------------
# The potential of cognitive-inspired neural network modeling framework for computer vision processing tasks
# Licensed under The MIT License [see LICENSE for details]
# Written by Guorun Li
# --------------------------------------------------------

# comparation models
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .smt import SMT
from .convnext import ConvNeXt
from .ResNet import resnetX
from .Transformer import ViT
from .VGG16 import VGG16
from .efficientnet_v2 import get_efficientnet_v2
from .densenet import DenseNet3
from .MobileNetv3 import MobileNetV3_Large, MobileNetV3_Small
from .Xception import Xception
from .PVT import PyramidVisionTransformer
from .gfnet import GFNet
from .spectformer_vanilla import SpectFormer_Vanilla
from .sequencer_vanilla import VanillaSequencer
from .vision_lstm.vision_lstm import VisionLSTM

# VCNU and VCM models
from .VCNNs_vcnu import VisionCognitiveNeuralUnits
from .VCNNs_vcm import VisionCognitiveModel

# PMB transfer learning models
from .vcnu_swin_transformer import VCNU_SwinTransformer
from .vcnu_smt import VCNU_SMT
from .vcnu_convnext import VCNU_ConvNeXt
from .attention import SelfAttentions
from .adapter_swin_transformer import Adapter_SwinTransformer
from .adapter_convnext import Adapter_ConvNeXt
from .adapter_smt import Adapter_SMT
def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'VCNN_VCNUs':
        model = VisionCognitiveNeuralUnits(
            filter_strategy1=config.MODEL.VCM.FILTER_STRATEGY1,
            filter_strategy2=config.MODEL.VCM.FILTER_STRATEGY2,

            use_layerscale=config.MODEL.VCM.USE_LAYERSCALE,
            layerscale_value=config.MODEL.VCM.LAYERSCALE_VALUE,
            image_size=config.DATA.IMG_SIZE,
            pretrain_image_size=config.MODEL.VCM.PRETRAIN_IMAGE_SIZE,

            patch_size=config.MODEL.VCM.PATCH_SIZE,
            embed_conv=config.MODEL.VCM.EMBED_CONV,
            in_chans=config.MODEL.VCM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.VCM.EMBED_DIM,
            depths=config.MODEL.VCM.DEPTHS,
            mlp_ratio=config.MODEL.VCM.MLP_RATIO,
            use_bias=config.MODEL.VCM.USE_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=layernorm,
            patch_norm=config.MODEL.VCM.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
            )
    elif model_type == 'VCNN_VCM':
        model = VisionCognitiveModel(
            filter_strategy1=config.MODEL.VCM.FILTER_STRATEGY1,
            filter_strategy2=config.MODEL.VCM.FILTER_STRATEGY2,
            
            use_layerscale=config.MODEL.VCM.USE_LAYERSCALE,
            layerscale_value=config.MODEL.VCM.LAYERSCALE_VALUE,
            image_size=config.DATA.IMG_SIZE,
            pretrain_image_size=config.MODEL.VCM.PRETRAIN_IMAGE_SIZE,
            
            patch_size=config.MODEL.VCM.PATCH_SIZE,
            embed_conv=config.MODEL.VCM.EMBED_CONV,
            in_chans=config.MODEL.VCM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.VCM.EMBED_DIM,
            depths=config.MODEL.VCM.DEPTHS,
            mlp_ratio=config.MODEL.VCM.MLP_RATIO,
            use_bias=config.MODEL.VCM.USE_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=layernorm,
            patch_norm=config.MODEL.VCM.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
            )
    
    elif model_type == 'SelfAttentions':
        model = SelfAttentions(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            norm_layer=layernorm,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS)
        
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)

    elif model_type == 'smt':
        model = SMT(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.VCNU_SMT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.VCNU_SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.VCNU_SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.VCNU_SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.VCNU_SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.VCNU_SMT.QKV_BIAS,
            qk_scale=config.MODEL.VCNU_SMT.QK_SCALE,
            use_layerscale=config.MODEL.VCNU_SMT.USE_LAYERSCALE,  #
            depths=config.MODEL.VCNU_SMT.DEPTHS,
            ca_attentions=config.MODEL.VCNU_SMT.CA_ATTENTIONS,
            head_conv=config.MODEL.VCNU_SMT.HEAD_CONV,
            expand_ratio=config.MODEL.VCNU_SMT.EXPAND_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'convnext':
        model = ConvNeXt(
            in_chans=config.MODEL.VCNU_CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VCNU_CONVNEXT.DEPTHS,
            dims=config.MODEL.VCNU_CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE,
            )

    elif model_type == 'resnet':
        model = resnetX(
            layers=config.MODEL.RESNET.LAYERS,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'vit':
        model = ViT(
            image_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            patches=config.MODEL.VIT.PATCHES,
            dim=config.MODEL.VIT.EMBED_DIM,
            ff_dim=config.MODEL.VIT.FEEDFORWARD_DIM,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            num_layers=config.MODEL.VIT.NUM_LAYERS,
            attention_dropout_rate=config.MODEL.VIT.ATTENTION_DROPOUT_RATE,
            dropout_rate=config.MODEL.VIT.DROPOUT_RATE,
            representation_size=config.MODEL.VIT.REPRESENTATION_SIZE,
            load_repr_layer=config.MODEL.VIT.LOAD_REPR_LAYER,
            classifier=config.MODEL.VIT.CLASSIFIER,
            positional_embedding=config.MODEL.VIT.POSITIONAL_EMBEDDING,
            in_channels=config.MODEL.VIT.IN_CHANS,

        )

    elif model_type == 'efficientnet_v2':
        model = get_efficientnet_v2(
                model_name=config.MODEL.EFFICIENTNET_V2.NAME,
                nclass=config.MODEL.NUM_CLASSES,
                dropout=config.MODEL.EFFICIENTNET_V2.DROPOUT,
                stochastic_depth=config.MODEL.EFFICIENTNET_V2.STOCHASTIC_DEPTH,
        )

    elif model_type == 'densenet201':
        model = DenseNet3(
                num_classes=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'vgg16':
        model = VGG16(
            num_labels=config.MODEL.NUM_CLASSES,
            img_size=config.DATA.IMG_SIZE,
        )

    elif model_type == 'mobilenetv3_l':
        model = MobileNetV3_Large(
            num_classes=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'mobilenetv3_s':
        model = MobileNetV3_Small(
            num_classes=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'xception':
        model = Xception(
            num_classes=config.MODEL.NUM_CLASSES,
        )

    elif model_type == 'pvt':
        model = PyramidVisionTransformer(
                img_size=config.DATA.IMG_SIZE, 
                patch_size=config.MODEL.PYT.PATCH_SIZE, 
                in_chans=config.MODEL.PYT.IN_CHANS, 
                num_classes=config.MODEL.NUM_CLASSES, 
                embed_dims=config.MODEL.PYT.EMBED_DIMS,
                num_heads=config.MODEL.PYT.NUM_HEADS, 
                mlp_ratios=config.MODEL.PYT.MLP_RATIOS, 
                qkv_bias=config.MODEL.PYT.QKV_BIAS, 
                qk_scale=config.MODEL.PYT.QK_SCALE, 
                drop_rate=config.MODEL.PYT.DROP_RATE,
                attn_drop_rate=config.MODEL.PYT.ATTN_DROP_RATE, 
                drop_path_rate=config.MODEL.PYT.DROP_PATH_RATE, 
                # norm_layer=nn.LayerNorm,
                depths=config.MODEL.PYT.DEPTHS, 
                sr_ratios=config.MODEL.PYT.SR_RATIOS, 
                num_stages=config.MODEL.PYT.NUM_STAGES, 
        )

    elif model_type == 'gfnet':
        model = GFNet(
                img_size=config.DATA.IMG_SIZE, 
                patch_size=config.MODEL.GFNET.PATCH_SIZE, 
                in_chans=config.MODEL.GFNET.IN_CHANS,
                num_classes=config.MODEL.NUM_CLASSES,
                embed_dim=config.MODEL.GFNET.EMBED_DIM, 
                depth=config.MODEL.GFNET.DEPTH, 
                mlp_ratio=config.MODEL.GFNET.MLP_RATIO,
                drop_path_rate=config.MODEL.GFNET.DROP_PATH_RATE,
        )
    elif model_type == 'spectformer_vanilla':
        model = SpectFormer_Vanilla(
                in_chans=config.MODEL.SPECTFORMER.IN_CHANS, 
                num_classes=config.MODEL.NUM_CLASSES, 
                stem_hidden_dim = config.MODEL.SPECTFORMER.STEM_HIDDEN_DIM,
                embed_dims=config.MODEL.SPECTFORMER.EMBED_DIMS,
                num_heads=config.MODEL.SPECTFORMER.NUM_HEADS, 
                mlp_ratios=config.MODEL.SPECTFORMER.MLP_RATIOS, 
                drop_path_rate=config.MODEL.SPECTFORMER.DROP_PATH_RATE, 
                depths=config.MODEL.SPECTFORMER.DEPTHS, 
                sr_ratios=config.MODEL.SPECTFORMER.SR_RATIOS, 
                num_stages=config.MODEL.SPECTFORMER.NUM_STAGES,
                token_label=config.MODEL.SPECTFORMER.TOKEN_LABEL,
        )
    elif model_type == 'vanillasequencer':
        model = VanillaSequencer(
                num_classes=config.MODEL.NUM_CLASSES,
                img_size=config.DATA.IMG_SIZE,
                in_chans=config.MODEL.VANILLA_SEQUENCER.IN_CHANS,
                layers=config.MODEL.VANILLA_SEQUENCER.LAYERS,
                patch_sizes=config.MODEL.VANILLA_SEQUENCER.PATCH_SIZES,
                embed_dims=config.MODEL.VANILLA_SEQUENCER.EMBED_DIMS,
                hidden_sizes=config.MODEL.VANILLA_SEQUENCER.HIDDEN_SIZES,
                mlp_ratios=config.MODEL.VANILLA_SEQUENCER.MLP_RATIOS,
                bidirectional=config.MODEL.VANILLA_SEQUENCER.BIDIRECTIONAL,
                shuffle=config.MODEL.VANILLA_SEQUENCER.SHUFFLE,
                ape=config.MODEL.VANILLA_SEQUENCER.APE,
                drop_path_rate=config.MODEL.VANILLA_SEQUENCER.DROP_PATH_RATE,
        )
    elif model_type == 'vil':
        model = VisionLSTM(
                dim=config.MODEL.VIL.DIM,
                input_shape=config.MODEL.VIL.INPUT_SHAPE,
                patch_size=config.MODEL.VIL.PATCH_SIZES,
                depth=config.MODEL.VIL.DEPTH,
                output_shape=(config.MODEL.NUM_CLASSES,),
                drop_path_rate=config.MODEL.VIL.DROP_PATH_RATE,
                stride=config.MODEL.VIL.STRIDE,
                legacy_norm=False,
        )

    # transfer learning
    elif model_type == 'vcnu_smt':
        model = VCNU_SMT(
            use_memory_embedding=config.MODEL.VCNU_SMT.USE_MEMORY_EMBEDDING,
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
            filter_strategy1=config.MODEL.VCNU_SMT.FILTER_STRATEGY1,
            filter_strategy2=config.MODEL.VCNU_SMT.FILTER_STRATEGY2,
            patch_size=config.MODEL.VCNU_SMT.PATCH_SIZE,
            ab_norm_attn=config.MODEL.VCNU_SMT.AB_NORM_ATTN,
            ab_norm_ltm=config.MODEL.VCNU_SMT.AB_NORM_LTM,
            model_style=config.MODEL.VCNU_SMT.MODEL_STYLE,

            use_layerscale=config.MODEL.VCNU_SMT.USE_LAYERSCALE,
            layerscale_value=config.MODEL.VCNU_SMT.LAYERSCALE_VALUE,  #

            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.VCNU_SMT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.VCNU_SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.VCNU_SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.VCNU_SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.VCNU_SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.VCNU_SMT.QKV_BIAS,
            qk_scale=config.MODEL.VCNU_SMT.QK_SCALE,
            depths=config.MODEL.VCNU_SMT.DEPTHS,
            ca_attentions=config.MODEL.VCNU_SMT.CA_ATTENTIONS,
            head_conv=config.MODEL.VCNU_SMT.HEAD_CONV,
            expand_ratio=config.MODEL.VCNU_SMT.EXPAND_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'adapter_smt':
        model = Adapter_SMT(
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
            filter_strategy1=config.MODEL.VCNU_SMT.FILTER_STRATEGY1,
            filter_strategy2=config.MODEL.VCNU_SMT.FILTER_STRATEGY2,
            patch_size=config.MODEL.VCNU_SMT.PATCH_SIZE,
            ab_norm_attn=config.MODEL.VCNU_SMT.AB_NORM_ATTN,
            ab_norm_ltm=config.MODEL.VCNU_SMT.AB_NORM_LTM,
            model_style=config.MODEL.VCNU_SMT.MODEL_STYLE,

            use_layerscale=config.MODEL.VCNU_SMT.USE_LAYERSCALE,
            layerscale_value=config.MODEL.VCNU_SMT.LAYERSCALE_VALUE,  #

            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.VCNU_SMT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.VCNU_SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.VCNU_SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.VCNU_SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.VCNU_SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.VCNU_SMT.QKV_BIAS,
            qk_scale=config.MODEL.VCNU_SMT.QK_SCALE,
            depths=config.MODEL.VCNU_SMT.DEPTHS,
            ca_attentions=config.MODEL.VCNU_SMT.CA_ATTENTIONS,
            head_conv=config.MODEL.VCNU_SMT.HEAD_CONV,
            expand_ratio=config.MODEL.VCNU_SMT.EXPAND_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'vcnu_convnext':
        model = VCNU_ConvNeXt(
            use_memory_embedding=config.MODEL.VCNU_CONVNEXT.USE_MEMORY_EMBEDDING,
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
            filter_strategy1=config.MODEL.VCNU_CONVNEXT.FILTER_STRATEGY1,
            filter_strategy2=config.MODEL.VCNU_CONVNEXT.FILTER_STRATEGY2,
            patch_size=config.MODEL.VCNU_CONVNEXT.PATCH_SIZE,
            img_size=config.DATA.IMG_SIZE,
            ab_norm_attn=config.MODEL.VCNU_CONVNEXT.AB_NORM_ATTN,
            ab_norm_ltm=config.MODEL.VCNU_CONVNEXT.AB_NORM_LTM,
            model_style=config.MODEL.VCNU_CONVNEXT.MODEL_STYLE,

            in_chans=config.MODEL.VCNU_CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VCNU_CONVNEXT.DEPTHS,
            dims=config.MODEL.VCNU_CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        
    elif model_type == 'adapter_convnext':
        model = Adapter_ConvNeXt(
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
            filter_strategy1=config.MODEL.VCNU_CONVNEXT.FILTER_STRATEGY1,
            filter_strategy2=config.MODEL.VCNU_CONVNEXT.FILTER_STRATEGY2,
            patch_size=config.MODEL.VCNU_CONVNEXT.PATCH_SIZE,
            img_size=config.DATA.IMG_SIZE,
            ab_norm_attn=config.MODEL.VCNU_CONVNEXT.AB_NORM_ATTN,
            ab_norm_ltm=config.MODEL.VCNU_CONVNEXT.AB_NORM_LTM,
            model_style=config.MODEL.VCNU_CONVNEXT.MODEL_STYLE,

            in_chans=config.MODEL.VCNU_CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VCNU_CONVNEXT.DEPTHS,
            dims=config.MODEL.VCNU_CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)


    elif model_type == 'vcnu_swin':
        model = VCNU_SwinTransformer(use_memory_embedding=config.MODEL.VCNU_SWIN.USE_MEMORY_EMBEDDING,
                                     is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
                                     filter_strategy1=config.MODEL.VCNU_SWIN.FILTER_STRATEGY1,
                                     filter_strategy2=config.MODEL.VCNU_SWIN.FILTER_STRATEGY2,
                                     pretrain_image_size=config.MODEL.VCNU_SWIN.PRETRAIN_IMAGE_SIZE,
                                     ab_norm_attn=config.MODEL.VCNU_SWIN.AB_NORM_ATTN,
                                     ab_norm_ltm=config.MODEL.VCNU_SWIN.AB_NORM_LTM,
                                     model_style=config.MODEL.VCNU_SWIN.MODEL_STYLE,
                                     training_mode=config.MODEL.VCNU_SWIN.TRAINING_MODE,
                                     use_layerscales=config.MODEL.VCNU_SWIN.USE_LAYERSCALE,
                                     layer_scale_init_value=config.MODEL.VCNU_SWIN.LAYER_SCALE_INIT_VALUE,

                                     img_size=config.DATA.IMG_SIZE,
                                     patch_size=config.MODEL.VCNU_SWIN.PATCH_SIZE,
                                     # out_dim=config.MODEL.VCNU_SWIN.OUT_DIM,
                                     in_chans=config.MODEL.VCNU_SWIN.IN_CHANS,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     embed_dim=config.MODEL.VCNU_SWIN.EMBED_DIM,
                                     depths=config.MODEL.VCNU_SWIN.DEPTHS,
                                     num_heads=config.MODEL.VCNU_SWIN.NUM_HEADS,
                                     window_size=config.MODEL.VCNU_SWIN.WINDOW_SIZE,
                                     mlp_ratio=config.MODEL.VCNU_SWIN.MLP_RATIO,
                                     qkv_bias=config.MODEL.VCNU_SWIN.QKV_BIAS,
                                     qk_scale=config.MODEL.VCNU_SWIN.QK_SCALE,
                                     drop_rate=config.MODEL.DROP_RATE,
                                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                     ape=config.MODEL.VCNU_SWIN.APE,
                                     norm_layer=layernorm,
                                     patch_norm=config.MODEL.VCNU_SWIN.PATCH_NORM,
                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                     fused_window_process=config.FUSED_WINDOW_PROCESS)
        
    elif model_type == 'adapter_swin':
        model = Adapter_SwinTransformer(is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
                                     filter_strategy1=config.MODEL.VCNU_SWIN.FILTER_STRATEGY1,
                                     filter_strategy2=config.MODEL.VCNU_SWIN.FILTER_STRATEGY2,
                                     pretrain_image_size=config.MODEL.VCNU_SWIN.PRETRAIN_IMAGE_SIZE,
                                     ab_norm_attn=config.MODEL.VCNU_SWIN.AB_NORM_ATTN,
                                     ab_norm_ltm=config.MODEL.VCNU_SWIN.AB_NORM_LTM,
                                     model_style=config.MODEL.VCNU_SWIN.MODEL_STYLE,
                                     training_mode=config.MODEL.VCNU_SWIN.TRAINING_MODE,
                                     use_layerscales=config.MODEL.VCNU_SWIN.USE_LAYERSCALE,
                                     layer_scale_init_value=config.MODEL.VCNU_SWIN.LAYER_SCALE_INIT_VALUE,

                                     img_size=config.DATA.IMG_SIZE,
                                     patch_size=config.MODEL.VCNU_SWIN.PATCH_SIZE,
                                     # out_dim=config.MODEL.VCNU_SWIN.OUT_DIM,
                                     in_chans=config.MODEL.VCNU_SWIN.IN_CHANS,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     embed_dim=config.MODEL.VCNU_SWIN.EMBED_DIM,
                                     depths=config.MODEL.VCNU_SWIN.DEPTHS,
                                     num_heads=config.MODEL.VCNU_SWIN.NUM_HEADS,
                                     window_size=config.MODEL.VCNU_SWIN.WINDOW_SIZE,
                                     mlp_ratio=config.MODEL.VCNU_SWIN.MLP_RATIO,
                                     qkv_bias=config.MODEL.VCNU_SWIN.QKV_BIAS,
                                     qk_scale=config.MODEL.VCNU_SWIN.QK_SCALE,
                                     drop_rate=config.MODEL.DROP_RATE,
                                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                     ape=config.MODEL.VCNU_SWIN.APE,
                                     norm_layer=layernorm,
                                     patch_norm=config.MODEL.VCNU_SWIN.PATCH_NORM,
                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                     fused_window_process=config.FUSED_WINDOW_PROCESS)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
