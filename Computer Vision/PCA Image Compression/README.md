# Image Compression through PCA

I use a very straightforward way to compress a grey-scale bitmap image through PCA.

The idea is to compress each row of pixels to a lower dimension. Of course, this is not an efficient method, and the result will not be very satisfactory (especially for huge images). I just want to demo the use of PCA.

---

> **Original Image**
>
> size: 7.62MB
>
> <img src="./data/img_gray.bmp" style="zoom:15%;" />

### Following are some reconstructed images (w.r.t. different amount of principle components)

> **16 Principle Components**
>
> size: 376KB
>
> <img src="./data/img_16pc.bmp" style="zoom:15%;" />

> **64 Principle Components**
>
> size: 1.41MB
>
> <img src="./data/img_64pc.bmp" style="zoom:15%;" />

> **128 Principle Components**
>
> size: 2.8MB
>
> <img src="./data/img_128pc.bmp" style="zoom:15%;" />

> **256 Principle Components**
>
> size: 5.59MB
>
> <img src="./data/img_256pc.bmp" style="zoom:15%;" />