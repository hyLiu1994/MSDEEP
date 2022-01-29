# optimizer
optimizer=dict(type='NoiseAdam', lr=0.1, weight_decay=0.0001, l2_beta=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)
