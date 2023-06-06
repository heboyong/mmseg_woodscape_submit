_base_ = './gta2cs_uda_stdc2_daformer.py'
model = dict(backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2')))
