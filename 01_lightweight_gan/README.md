# Training and Using the Lightweight GAN

The easiest part of the pipeline that just so happens to take the longest.

For me, I ran the following command:

```bash
lightweight_gan --data /home/nathancooperjones/GitHub/apebase/ipfs_jpg/ --name lightweight_ape_gan --image-size 512 --batch-size 16 --gradient-accumulate-every 4 --num-train-steps 300000 --aug-prob 0.35 --aug-types [cutout,color] --attn-res-layers [32,64]
```

If you instead would like to start with my pre-trained weights, you can download those [here](https://thisboredapedoesnotexist.s3.amazonaws.com/models/lightweight_ape_gan.zip). With these weights, you can follow the directions on [lucidrains' source repo here](https://github.com/lucidrains/lightweight-gan) to generate images, with something similar to:

```bash
lightweight_gan --name lightweight_ape_gan --load-from 256 --generate --num-image-tiles 5
```

Good luck!
