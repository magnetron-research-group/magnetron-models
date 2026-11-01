# magnetron-models
Model architectures and inference pipelines implemented using the [Magnetron](https://github.com/MarioSieg/magnetron) ML framework.

## Models
| Category | Models |
|----------|--------|
| Language Models | Qwen 3, Qwen 3.5, Qwen 3.8 |
| Image Generation | Qwen-Image 2.1 |
| Speech Recognition | Whisper |

## Weight Conversion
Magnetron stores data in it's own **MAG** (.mag) snapshot files.
Pretrained weights from HF are converted to MAG files using the [Magnetron-Converters](https://github.com/magnetron-research-group/magnetron-converters) repo.
This is usually not needed because we store [pre-converted weights on HF](https://huggingface.co/mario-sieg) for all the models.
