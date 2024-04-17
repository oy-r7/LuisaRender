//
// Created by mike on 4/17/24.
//
/**
* # bump_image = np.power(
            #     cv.imread(material_bump, cv.IMREAD_GRAYSCALE) / 255, 2.2
            # )
            # h, w = bump_image.shape[:2]
            # scale = 4
            # bump_image = cv.resize(bump_image, (w * scale, h * scale), interpolation=cv.INTER_LANCZOS4)
            # bump_image = cv.GaussianBlur(bump_image, (5, 5), 0)
            # dx_image = cv.copyMakeBorder(bump_image, 0, 0, 1, 1, cv.BORDER_REPLICATE)
            # dy_image = cv.copyMakeBorder(bump_image, 1, 1, 0, 0, cv.BORDER_REPLICATE)
            # strength = min(w, h) / 50 * scale
            # dx_image = np.clip(strength * (dx_image[:, 2:] - dx_image[:, :-2]), -5, 5)
            # dy_image = np.clip(-strength * (dy_image[2:, :] - dy_image[:-2, :]), -5, 5)
            # dx_image = cv.resize(dx_image, (w, h), interpolation=cv.INTER_AREA)
            # dy_image = cv.resize(dy_image, (w, h), interpolation=cv.INTER_AREA)
            # dz_image = np.ones_like(dx_image)
            # norm = np.sqrt(dx_image**2 + dy_image**2 + dz_image**2)
            # normal_image = np.dstack([dz_image, dy_image, dx_image])
            # normal_image = (normal_image / norm[:, :, np.newaxis]) * 0.5 + 0.5
            # # normal_image = cv.GaussianBlur(normal_image, (3, 3), 0)
            # # normal_image = cv.resize(normal_image, (w, h), interpolation=cv.INTER_CUBIC)
            # normal_image = np.uint8(np.clip(normal_image * 255, 0, 255))
            # save_name = f"lr_exported_textures/{material_bump.split('/')[-1]}"
            # cv.imwrite(save_name, normal_image)
*/