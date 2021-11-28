# This Bored Ape Does Not Exist

![](https://user-images.githubusercontent.com/31417712/143785778-69d93e1a-0210-4b8d-8f7a-d283989a0e4c.gif)

Code to fully reproduce the results for the blog post [These Bored Apes Do Not Exist]().

View results at [thisboredapedoesnotexist.nathancooperjones.com](https://thisboredapedoesnotexist.nathancooperjones.com/).

## Getting Started
On a machine with Python installed, first install requirements with:

```bash
pip install -r requirements.in
```

From here, you can take one of two paths:

1) **Training a model from scratch.** To do this, you'll need some data. With a free GPU available and plenty of time, head on over to [``00_get_data``](00_get_data) to get the dataset needed for the GAN. From there, you'll head to [``01_lightweight_gan``](01_lightweight_gan), [``02_super_resolution``](02_super_resolution), and [``03_resizing_and_png_conversion``](03_resizing_and_png_conversion), in that order.

2) **Running pre-trained models in inference-mode.** If you do not want to train any models, but instead just run all models in inference mode, then head over to [``01_lightweight_gan``](01_lightweight_gan) to find the URL to download the model weights. From there, you'll head to [``02_super_resolution``](02_super_resolution) and [``03_resizing_and_png_conversion``](03_resizing_and_png_conversion), in that order.
