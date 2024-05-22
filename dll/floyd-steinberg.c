#include <stdio.h>
#include <stdint.h>


uint8_t clampTo8Bit(int v) {
    if ((v & ~0xFF) != 0) {
        v = ((~v) >> 31) & 0xFF;
    }
    return v;
}


void find_closest(uint8_t pixel[3], uint8_t palette[][3], int plt_size) {
    int prev_distance = 300000;
    int ind = 0;
    for (int k=0; k<plt_size; k++) {
        int distance = 0;
        for (int i=0; i<3; i++) {
            int diff = palette[k][i]-pixel[i];
            distance += diff*diff;
        }
        if (distance < prev_distance) {
            prev_distance = distance;
            ind = k;
        }
    }
    for (int j=0; j<3; j++) {
        pixel[j] = palette[ind][j];
    }
}

unsigned char* dither(unsigned char *img, int w, int h, int channels, uint8_t palette[][3], int plt_size,
            int tmp, float tmp2) {
    if(img == NULL) {
        printf("Error loading the image\n");
        return '\0';
    }

    size_t img_size = w * h * channels;
    uint8_t new_pixel[3];

    for(int y = 0; y < h-1; y++) {
        for(int x = 1; x < w-1; x++) {
            unsigned char *src_pixel = img + (y * w + x) * 4;
            unsigned char *src_pixel1 = img + (y * w + (x+1)) * 4;
            unsigned char *src_pixel2 = img + ((y+1) * w + x) * 4;
            unsigned char *src_pixel3 = img + ((y+1) * w + (x-1)) * 4;
            unsigned char *src_pixel4 = img + ((y+1) * w + (x+1)) * 4;
            for (int n=0; n<3; n++) {
                new_pixel[n] = src_pixel[n];
            }
            find_closest(new_pixel, palette, plt_size);
            for (int i=0; i<3; i++) {
                float error = src_pixel[i]-new_pixel[i];
                src_pixel[i] = new_pixel[i];
                src_pixel1[i] = clampTo8Bit(src_pixel1[i]+7*error/16);
                src_pixel2[i] = clampTo8Bit(src_pixel2[i]+5*error/16);
                src_pixel3[i] = clampTo8Bit(src_pixel3[i]+3*error/16);
                src_pixel4[i] = clampTo8Bit(src_pixel4[i]+error/16);
            }
        }
    }

    return img;
}


int main() {
}