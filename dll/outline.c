#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>


uint8_t clampTo8Bit(int v) {
    if ((v & ~0xFF) != 0) {
        v = ((~v) >> 31) & 0xFF;
    }
    return v;
}


unsigned char* dither(unsigned char *img, int w, int h, int channels, int threshold, short outlineColor[4], int thickness, char outlineType) {
    if(img == NULL) {
        printf("Error loading the image\n");
        return '\0';
    }
    size_t img_size = w * h * channels;
    unsigned char *out_img = malloc(img_size);
    unsigned char *final_img = malloc(img_size);
    float distance; float circleBias = 1.15;
    if (outlineType == 'c') {
        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                unsigned char *src_pixel = img + (y * w + x) * 4;
                unsigned char *dst_pixel = out_img + (y * w + x) * 4;
                if (*(src_pixel+3) >= threshold) {
                    for (int i=-thickness/2; i<=thickness/2; i++) {
                        for (int j=-thickness/2; j<=thickness/2; j++) {
                            int yy = y + i; int xx = x + j;
                            if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                                distance = (yy-y)*(yy-y) + (xx-x)*(xx-x);
                                if (distance <= circleBias*(thickness/2+1)*(thickness/2+1)) { unsigned char *dst_pixel = out_img + (yy * w + xx) * 4; for (int k = 0; k < 4; k++) { dst_pixel[k] = outlineColor[k]; } } }
                        }
                    }
                }
            }

        }
    } else if (outlineType == 'q') {
        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                unsigned char *src_pixel = img + (y * w + x) * 4;
                unsigned char *dst_pixel = out_img + (y * w + x) * 4;
                if (*(src_pixel+3) >= threshold) {
                    for (int i=-thickness/2; i<=thickness/2; i++) {
                        for (int j=-thickness/2; j<=thickness/2; j++) {
                            int yy = y + i; int xx = x + j;
                            if (yy >= 0 && yy < h && xx >= 0 && xx < w) { unsigned char *dst_pixel = out_img + (yy * w + xx) * 4; for (int k = 0; k < 4; k++) { dst_pixel[k] = outlineColor[k]; } }
                        }
                    }
                }
            }

        }
    } else if (outlineType == 's') {
        for(int y = thickness; y < h-thickness; y++) {
            for(int x = thickness; x < w-thickness; x++) {
                unsigned char *src_pixel = img + (y * w + x) * 4;
                unsigned char *dst_pixel = out_img + (y * w + x) * 4;
                if (*(src_pixel+3) >= threshold) {
                    for (int i=-thickness/2; i<=thickness/2; i++) {
                        for (int j=-thickness/2; j<=thickness/2; j++) {
                            int yy = y + i; int xx = x + j;
                            if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                                distance = abs(yy-y)+abs(xx-x);
                                if (distance <= thickness/2 + 1) { unsigned char *dst_pixel = out_img + (yy * w + xx) * 4; for (int k = 0; k < 4; k++) { dst_pixel[k] = outlineColor[k]; } } }
                        }
                    }
                }
            }

        }
    }
    for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                unsigned char *org_pixel = img + (y * w + x) * 4;
                unsigned char *src_pixel = out_img + (y * w + x) * 4;
                unsigned char *dst_pixel = final_img + (y * w + x) * 4;
                if (src_pixel[3] >= threshold && org_pixel[3] >= threshold) {
                    for(int k=0; k<4; k++) {
                        dst_pixel[k] = org_pixel[k];
                    }
                } else {
                    for(int k=0; k<4; k++) {
                        dst_pixel[k] = src_pixel[k];
                    }
                }
            }
    }
    return final_img;

}


int main() {
}