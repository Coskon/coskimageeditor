#include <stdio.h>
#include <stdint.h>

const int BAYER_2X2[][2] = {{0, 2}, {3, 1}};
const int BAYER_4X4[][4] = {{0,  8,  2, 10},
                            {12, 4, 14,  6},
                            {3, 11,  1,  9},
                            {15, 7, 13,  5}};
const int BAYER_8X8[][8] = {{ 0, 32,  8, 40,  2, 34, 10, 42},
                            {48, 16, 56, 24, 50, 18, 58, 26},
                            {12, 44,  4, 36, 14, 46,  6, 38},
                            {60, 28, 52, 20, 62, 30, 54, 22},
                            { 3, 35, 11, 43,  1, 33,  9, 41},
                            {51, 19, 59, 27, 49, 17, 57, 25},
                            {15, 47,  7, 39, 13, 45,  5, 37},
                            {63, 31, 55, 23, 61, 29, 53, 21}};
const int BAYER_16X16[][16] = {{  0, 192,  48, 240,  12, 204,  60, 252,   3, 195,  51, 243,  15, 207,  63, 255},
                               {128,  64, 176, 112, 140,  76, 188, 124, 131,  67, 179, 115, 143,  79, 191, 127},
                               { 32, 224,  16, 208,  44, 236,  28, 220,  35, 227,  19, 211,  47, 239,  31, 223},
                               {160,  96, 144,  80, 172, 108, 156,  92, 163,  99, 147,  83, 175, 111, 159,  95},
                               {  8, 200,  56, 248,   4, 196,  52, 244,  11, 203,  59, 251,   7, 199,  55, 247},
                               {136,  72, 184, 120, 132,  68, 180, 116, 139,  75, 187, 123, 135,  71, 183, 119},
                               { 40, 232,  24, 216,  36, 228,  20, 212,  43, 235,  27, 219,  39, 231,  23, 215},
                               {168, 104, 152,  88, 164, 100, 148,  84, 171, 107, 155,  91, 167, 103, 151,  87},
                               {  2, 194,  50, 242,  14, 206,  62, 254,   1, 193,  49, 241,  13, 205,  61, 253},
                               {130,  66, 178, 114, 142,  78, 190, 126, 129,  65, 177, 113, 141,  77, 189, 125},
                               { 34, 226,  18, 210,  46, 238,  30, 222,  33, 225,  17, 209,  45, 237,  29, 221},
                               {162,  98, 146,  82, 174, 110, 158,  94, 161,  97, 145,  81, 173, 109, 157,  93},
                               { 10, 202,  58, 250,   6, 198,  54, 246,   9, 201,  57, 249,   5, 197,  53, 245},
                               {138,  74, 186, 122, 134,  70, 182, 118, 137,  73, 185, 121, 133,  69, 181, 117},
                               { 42, 234,  26, 218,  38, 230,  22, 214,  41, 233,  25, 217,  37, 229,  21, 213},
                               {170, 106, 154,  90, 166, 102, 150,  86, 169, 105, 153,  89, 165, 101, 149,  85}};
const int BAYER_32X32[][32] = {{0, 512, 640, 128, 2, 130, 514, 642, 8, 520, 136, 648, 10, 522, 138, 650, 32, 544, 160, 672, 34, 546, 162, 674, 40, 552, 168, 680, 42, 554, 170, 682},
                               {768, 256, 384, 896, 770, 258, 898, 386, 776, 264, 904, 392, 778, 266, 906, 394, 800, 288, 928, 416, 802, 290, 930, 418, 808, 296, 936, 424, 810, 298, 938, 426},
                               {192, 576, 64, 704, 66, 194, 706, 578, 200, 712, 72, 584, 202, 714, 74, 586, 224, 736, 96, 608, 226, 738, 98, 610, 232, 744, 104, 616, 234, 746, 106, 618},
                               {960, 320, 832, 448, 962, 450, 834, 322, 968, 456, 840, 328, 970, 458, 842, 330, 992, 480, 864, 352, 994, 482, 866, 354, 1000, 488, 872, 360, 1002, 490, 874, 362},
                               {16, 528, 144, 656, 18, 530, 146, 658, 536, 664, 24, 152, 26, 538, 154, 666, 48, 560, 178, 176, 688, 50, 562, 690, 568, 696, 56, 184, 58, 570, 186, 698},
                               {784, 272, 912, 400, 786, 274, 914, 402, 280, 408, 792, 920, 794, 282, 922, 410, 816, 304, 306, 944, 432, 818, 946, 434, 312, 440, 824, 952, 826, 314, 954, 442},
                               {208, 720, 80, 592, 210, 722, 82, 594, 600, 728, 88, 216, 120, 218, 730, 90, 248, 602, 762, 122, 240, 752, 754, 112, 624, 760, 242, 114, 632, 626, 250, 634},
                               {976, 464, 848, 336, 978, 466, 850, 338, 344, 472, 1016, 984, 856, 986, 474, 858, 346, 504, 378, 1008, 496, 1010, 498, 880, 368, 888, 882, 376, 370, 1018, 506, 890},
                               {4, 516, 132, 644, 6, 518, 134, 646, 140, 12, 524, 652, 14, 526, 142, 654, 36, 548, 164, 676, 38, 550, 166, 678, 44, 556, 172, 684, 46, 558, 174, 686},
                               {772, 260, 900, 388, 774, 262, 902, 390, 908, 780, 268, 396, 782, 270, 910, 398, 804, 292, 932, 420, 806, 294, 934, 422, 812, 300, 940, 428, 814, 302, 942, 430},
                               {196, 708, 68, 580, 198, 710, 70, 582, 716, 204, 588, 76, 206, 718, 78, 590, 228, 740, 100, 612, 230, 742, 102, 614, 236, 748, 108, 620, 238, 750, 110, 622},
                               {964, 452, 836, 324, 966, 454, 838, 326, 460, 972, 332, 844, 974, 462, 846, 334, 996, 484, 868, 356, 998, 486, 870, 358, 1004, 492, 876, 364, 1006, 494, 878, 366},
                               {20, 532, 148, 660, 22, 534, 150, 662, 668, 28, 156, 540, 30, 542, 158, 670, 52, 564, 180, 692, 54, 566, 182, 694, 700, 60, 62, 574, 190, 188, 572, 702},
                               {916, 276, 404, 788, 790, 278, 918, 406, 796, 924, 284, 412, 798, 286, 926, 414, 820, 308, 948, 436, 822, 310, 950, 438, 828, 316, 318, 958, 444, 956, 830, 446},
                               {764, 638, 766, 598, 212, 724, 84, 596, 254, 214, 726, 86, 732, 92, 220, 604, 222, 734, 94, 606, 124, 252, 244, 756, 116, 628, 246, 758, 118, 636, 630, 126},
                               {382, 892, 894, 1022, 340, 980, 468, 852, 982, 470, 854, 342, 860, 988, 348, 476, 990, 478, 862, 350, 508, 1012, 500, 884, 372, 380, 1014, 502, 886, 1020, 374, 510},
                               {1, 513, 515, 3, 131, 643, 129, 641, 9, 521, 11, 523, 139, 651, 137, 649, 33, 545, 35, 547, 163, 675, 161, 673, 41, 553, 43, 555, 171, 683, 169, 681},
                               {769, 257, 259, 771, 899, 387, 897, 385, 777, 265, 779, 267, 907, 395, 905, 393, 801, 289, 803, 291, 931, 419, 929, 417, 809, 297, 811, 299, 939, 427, 937, 425},
                               {193, 705, 579, 67, 195, 707, 65, 577, 201, 713, 203, 715, 75, 587, 73, 585, 225, 737, 227, 739, 99, 611, 97, 609, 233, 745, 235, 747, 107, 619, 105, 617},
                               {833, 961, 449, 323, 835, 963, 451, 321, 969, 457, 971, 459, 843, 331, 841, 329, 993, 481, 995, 483, 867, 355, 865, 353, 1001, 489, 1003, 491, 875, 363, 873, 361},
                               {17, 529, 19, 531, 147, 659, 145, 657, 25, 537, 27, 155, 667, 539, 153, 665, 177, 49, 561, 563, 691, 51, 179, 689, 57, 569, 187, 59, 699, 571, 185, 697},
                               {785, 273, 787, 275, 915, 403, 913, 401, 793, 281, 795, 923, 283, 411, 921, 409, 945, 817, 307, 435, 819, 947, 305, 433, 825, 313, 955, 315, 827, 443, 953, 441},
                               {633, 209, 721, 211, 723, 83, 595, 81, 593, 217, 729, 603, 91, 219, 731, 89, 601, 121, 123, 251, 241, 753, 115, 627, 243, 755, 113, 625, 249, 761, 635, 763},
                               {891, 1019, 977, 465, 979, 467, 851, 339, 849, 337, 985, 473, 859, 987, 347, 475, 889, 857, 345, 377, 505, 497, 1009, 883, 371, 1011, 499, 507, 881, 369, 379, 1017},
                               {5, 517, 7, 519, 135, 647, 133, 645, 13, 141, 143, 655, 527, 15, 525, 653, 37, 549, 39, 551, 167, 679, 165, 677, 45, 557, 47, 559, 175, 687, 173, 685},
                               {773, 261, 775, 263, 903, 391, 901, 389, 269, 781, 911, 399, 271, 783, 909, 397, 805, 293, 807, 295, 935, 423, 933, 421, 813, 301, 815, 303, 943, 431, 941, 429},
                               {197, 709, 199, 711, 71, 583, 69, 581, 77, 205, 719, 591, 79, 207, 717, 589, 229, 741, 231, 743, 103, 615, 101, 613, 237, 749, 239, 751, 111, 623, 109, 621},
                               {965, 453, 967, 455, 839, 327, 837, 325, 973, 461, 463, 335, 847, 975, 845, 333, 997, 485, 999, 487, 871, 359, 869, 357, 1005, 493, 1007, 495, 879, 367, 877, 365},
                               {21, 533, 535, 663, 23, 151, 149, 661, 29, 541, 159, 543, 671, 31, 157, 669, 53, 565, 567, 695, 55, 183, 181, 693, 573, 61, 575, 703, 189, 63, 191, 701},
                               {789, 277, 407, 791, 919, 279, 917, 405, 797, 285, 287, 415, 799, 927, 925, 413, 821, 309, 439, 951, 823, 311, 949, 437, 829, 957, 319, 447, 831, 317, 959, 445},
                               {637, 639, 767, 85, 597, 213, 725, 599, 727, 119, 87, 247, 215, 221, 733, 607, 735, 95, 223, 93, 605, 759, 125, 245, 631, 757, 253, 117, 629, 127, 765, 255},
                               {511, 893, 1021, 895, 1023, 981, 469, 471, 855, 983, 343, 853, 341, 989, 477, 351, 479, 863, 375, 991, 861, 349, 887, 381, 1015, 1013, 501, 503, 885, 373, 509, 383}};

uint8_t clampTo8Bit(int v) {
    if ((v & ~0xFF) != 0) {
        v = ((~v) >> 31) & 0xFF;
    }
    return v;
}


void find_closest(int pixel[3], uint8_t palette[][3], int plt_size) {
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

unsigned char* dither(unsigned char *img, int w, int h, int channels, uint8_t palette[][3],
    int plt_size, int s, float strength) {
    if(img == NULL) {
        printf("Error loading the image\n");
        return '\0';
    }
    size_t img_size = w * h * channels;
    int new_pixel[3]; int bayer_value;
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
            unsigned char *src_pixel = img + (y * w + x) * 4;
            if (s == 2) { bayer_value = BAYER_2X2[y%2][x%2]; }
            if (s == 4) { bayer_value = BAYER_4X4[y%4][x%4]; }
            if (s == 8) { bayer_value = BAYER_8X8[y%8][x%8]; }
            if (s == 16) { bayer_value = BAYER_16X16[y%16][x%16]; }
            else { bayer_value = BAYER_32X32[y%32][x%32]; }
            for(int i = 0; i < 3; i++) {
                new_pixel[i] = clampTo8Bit(src_pixel[i]+0.5*(bayer_value/strength - 0.5));
            }
            find_closest(new_pixel, palette, plt_size);
            src_pixel[0] = new_pixel[0];
            src_pixel[1] = new_pixel[1];
            src_pixel[2] = new_pixel[2];
        }
    }

    return img;
}


int main() {
}