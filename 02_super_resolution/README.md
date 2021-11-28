# Training and Using the Super-Resolution Model

In order not to duplicate a whole repository's worth of code, I included the ``Waifu2x`` repo as a submodule in this directory.

In order to train a new model, move the ``custom_dataset.py`` and ``custom_training_script.py`` files into the ``Waifu2x`` directory, then call the training script with:

```bash
python custom_training_script
```

If you would instead like to skip out on training completely and download my pre-trained model, you can do so [here](https://thisboredapedoesnotexist.s3.amazonaws.com/models/waifu2x/CARN_model_checkpoint.pt).

To apply the trained super-resolution model to images, copy the ``super-resolution-apes.ipynb`` notebook into the ``Waifu2x`` directory, point the pre-trained model path to the one you just trained, and run from the top to the bottom.

Good luck!
