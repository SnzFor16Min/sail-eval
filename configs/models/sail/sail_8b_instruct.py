from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='sail_8b_instruct',
        path='terter',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
