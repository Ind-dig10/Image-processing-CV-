import cv2
import numpy as np
import matplotlib.pyplot as plt


def transform2D(image, affine_matrix):
    # grab the shape of the image
    B, H, W, C = image.shape
    M = affine_matrix

    # mesh grid generation
    # use x = np.linspace(-1, 1, W)  if you want to rotate about center
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    x_t, y_t = np.meshgrid(x, y)

    # augment the dimensions to create homogeneous coordinates
    # reshape to (xt, yt, 1)
    ones = np.ones(np.prod(x_t.shape))
    sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # repeat to number of batches
    sampling_grid = np.resize(sampling_grid, (B, 3, H * W))

    # transform the sampling grid, i.e. batch multiply
    batch_grids = np.matmul(M, sampling_grid)  # the batch grid has the shape (B, 2, H*W)

    # reshape to (B, H, W, 2)
    batch_grids = batch_grids.reshape(B, 2, H, W)
    batch_grids = np.moveaxis(batch_grids, 1, -1)

    # bilinear resampler
    x_s = batch_grids[:, :, :, 0:1].squeeze()
    y_s = batch_grids[:, :, :, 1:2].squeeze()

    # rescale x and y to [0, W/H]
    # use this function if you want to rotate about center
    # x = ((x_s+1.)*W)*0.5
    # y = ((y_s+1.)*H)*0.5
    x = ((x_s) * W)
    y = ((y_s) * H)

    # for each coordinate we need to grab the corner coordinates
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1

    # clip to fit actual image size
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # grab the pixel value for each corner coordinate
    Ia = image[np.arange(B)[:, None, None], y0, x0]
    Ib = image[np.arange(B)[:, None, None], y1, x0]
    Ic = image[np.arange(B)[:, None, None], y0, x1]
    Id = image[np.arange(B)[:, None, None], y1, x1]

    # calculated the weighted coefficients and actual pixel value
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=3)
    wb = np.expand_dims(wb, axis=3)
    wc = np.expand_dims(wc, axis=3)
    wd = np.expand_dims(wd, axis=3)

    # compute output
    image_out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    image_out = image_out.astype(np.int64)

    return image_out


def affine_transform(mode, image):
    if mode == '2D':
        # 2D
        input_img = image

        # define the affine matrix
        # initialize M to identity transform
        M = np.array([[1., 0., 0.], [0., 1., 0.]])
        # repeat num_batch times
        M = np.resize(M, (input_img.shape[0], 2, 3))

        # change affine matrix values
        # translation
        M[0, :, :] = [[1., 0., 0.], [0., 1., 0.]]
        img_translate = transform2D(input_img, M)

        # rotation
        angle = 45  # degree
        M[0, :, :] = [[math.cos(angle / 180 * math.pi), -math.sin(angle / 180 * math.pi), 0],
                      [math.sin(angle / 180 * math.pi), math.cos(angle / 180 * math.pi), 0]]
        img_rotate = transform2D(input_img, M)

        # shear
        M[0, :, :] = [[1, 0.5, 0], [0.5, 1, 0]]
        img_shear = transform2D(input_img, M)

        image_matching_metric(input_img[0, :, :, :], img_translate[0, :, :, :], title="Translate", plot=True)
        image_matching_metric(input_img[0, :, :, :], img_rotate[0, :, :, :], title="Rotate", plot=True)
        image_matching_metric(input_img[0, :, :, :], img_shear[0, :, :, :], title="Shear", plot=True)

        plt.figure(1)
        ax1 = plt.subplot(221)
        plt.imshow(input_img[0, :, :, :], cmap="gray")
        ax1.title.set_text('Original')
        plt.axis("off")

        ax2 = plt.subplot(222)
        plt.imshow(img_translate[0, :, :, :], cmap="gray")
        ax2.title.set_text('Translation')
        plt.axis("off")

        ax3 = plt.subplot(223)
        plt.imshow(img_rotate[0, :, :, :], cmap="gray")
        ax3.title.set_text('Rotation')
        plt.axis("off")

        ax4 = plt.subplot(224)
        plt.imshow(img_shear[0, :, :, :], cmap="gray")
        ax4.title.set_text('Shear')
        plt.axis("off")

        plt.show()

    else:
        # 3D
        layer_num = 8
        input_img = load3D(layer_num)

        # define the affine matrix
        # initialize M to identity transform
        M = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
        # repeat num_batch times
        M = np.resize(M, (input_img.shape[0], 3, 4))

        # change affine matrix values
        # translation
        M[0, :, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        img_translate = transform3D(input_img, M)

        # rotation
        alpha = 20  # degree
        beta = 0
        gamma = 0

        # convert from degree to radian
        alpha = alpha * math.pi / 180
        beta = beta * math.pi / 180
        gamma = gamma * math.pi / 180
        # Tait-Bryan angles in homogeneous form, reference: https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf
        Rx = [[1, 0, 0, 0], [0, math.cos(alpha), -math.sin(alpha), 0], [0, math.sin(alpha), math.cos(alpha), 0],
              [0., 0., 0, 1.]]
        Ry = [[math.cos(beta), 0, math.sin(beta), 0], [0, 1, 0, 0], [-math.sin(beta), 0, math.cos(beta), 0],
              [0., 0., 0., 1.]]
        Rz = [[math.cos(gamma), -math.sin(gamma), 0, 0], [math.sin(gamma), math.cos(gamma), 0, 0], [0, 0, 1, 0],
              [0., 0., 0., 1.]]

        print("Rx", Rx)
        print("Ry", Ry)
        print("Rz", Rz)

        M[0, :, :] = np.matmul(Rz, np.matmul(Ry, Rx))[0:3, :]
        print(M)

        img_rotate = transform3D(input_img, M)

        # shear
        M[0, :, :] = [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0]]
        img_shear = transform3D(input_img, M)

        image_matching_metric(input_img[0, :, :, :, :], img_translate[0, :, :, :, :], title="Translate", plot=False)
        image_matching_metric(input_img[0, :, :, :, :], img_rotate[0, :, :, :, :], title="Rotate", plot=False)
        image_matching_metric(input_img[0, :, :, :, :], img_shear[0, :, :, :, :], title="Shear", plot=False)

        fig = plt.figure(1)

        for layer in range(input_img.shape[3]):
            ax0 = fig.add_subplot(4, input_img.shape[3], layer + 1)
            ax0.imshow(input_img[0, :, :, layer, 0], cmap="gray")
            ax0.axis("off")

            ax1 = fig.add_subplot(4, input_img.shape[3], input_img.shape[3] * 1 + layer + 1)
            ax1.imshow(img_translate[0, :, :, layer, 0], cmap="gray")
            ax1.axis("off")

            ax2 = fig.add_subplot(4, input_img.shape[3], input_img.shape[3] * 2 + layer + 1)
            ax2.imshow(img_rotate[0, :, :, layer, 0], cmap="gray")
            ax2.axis("off")

            ax3 = fig.add_subplot(4, input_img.shape[3], input_img.shape[3] * 3 + layer + 1)
            ax3.imshow(img_shear[0, :, :, layer, 0], cmap="gray")
            ax3.axis("off")

        plt.show()


def affine(img, tx, ty):
    H, W, C = img.shape
    print(H)
    print(W)
    print(C)
    tem = img.copy()
    img = np.zeros((H + 2, W + 2, C), dtype=np.float32)
    img[1:H + 1, 1:W + 1] = tem

    H_new = np.round(H).astype(np.int)
    W_new = np.round(W).astype(np.int)
    out = np.zeros((H_new + 1, W_new + 1, C), dtype=np.float32)
    print(H_new)
    print(W_new)

    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
    print(x_new)
    print(y_new)

    x = np.round(x_new).astype(np.int) - tx
    y = np.round(y_new).astype(np.int) - ty
    print(x)
    print(y)

    x = np.minimum(np.maximum(x, 0), W + 1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H + 1).astype(np.int)

    out[y_new, x_new] = img[y, x]

    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)

    return out