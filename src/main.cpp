#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui_c.h"

#define PI 3.14

typedef struct
{
    double class_center;
    double pre_class_center;
    double cluster_mean;
    double element_count;
}CENTER;

int tag_value = 0;

int **rgb_to_h(int *rgb, int height, int width)
{
    int i, j, red, green, blue, hue_angle;
    double value;

    int **hueMatrix = (int**)calloc(height, sizeof(int*));
    for(i = 0; i < height; i++)
    {
		hueMatrix[i] =(int*) calloc(width, sizeof(int));
	}

	for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
		{
            blue = rgb[i*width*3 + j*3];
            green = rgb[i*width*3 + j*3 + 1];
            red = rgb[i*width*3 + j*3 + 2];
            value = (0.5 * ((red - green) + (red - blue))) / sqrt((red - green) * (red - green) + (red - blue) * (green - blue));

            if(value >= -1 && value <= 1)
                hue_angle =  ((acos(value) * 180.0 )/ PI);
            else
                hue_angle = 0;

            if (blue > green)
                hue_angle = 360 - hue_angle;
            hueMatrix[i][j] = hue_angle;
		}
	}
	return hueMatrix;
}

int **k_means(int **hueMatrix, int height, int width, int k_means_value)
{
    int i,j,k;
    int minimum, cluster;
    double value;

    int **k_meansMatrix = (int**)calloc(height, sizeof(int*));
    for(i = 0;i<height;i++)
	{
		k_meansMatrix[i] =(int*) calloc(width, sizeof(int));
	}

	CENTER *k_means_classes = (CENTER*)calloc(k_means_value, sizeof(CENTER));

	srand(time(0));

	for(i = 0; i < k_means_value; i++)
	{
	    int row = rand() % height;
	    int column = rand() % width;

	    k_means_classes[i].class_center = (double)hueMatrix[row][column];
	    printf("\nInitialize cluster centers: %lf\n",k_means_classes[i].class_center);
	}

	int check = 1;

	while( check == 1 )
	{
	    for(i = 0; i < k_means_value; i++)
	    {
	        //Initialize k_means classes values
            k_means_classes[i].pre_class_center = k_means_classes[i].class_center;
            k_means_classes[i].element_count = 0;
	    }
	    // compute the each pixels' proximity for k classes
	    for(i = 0; i < height; i++)
	    {
	        for(j = 0; j < width; j++)
	        {
	            minimum = 360;   // Initialize maximum Hue value
	            // Find the nearest pixel
	            for(k = 0; k < k_means_value; k++)
	            {
	                int difference = abs((double)hueMatrix[i][j] - k_means_classes[k].class_center);
                    if(difference < minimum)
                    {
                        minimum = difference;
                        cluster = k;
                    }
	            }
	            // calculate new value for k. class
	            double pre_mean = (k_means_classes[cluster].cluster_mean * k_means_classes[cluster].element_count) / (k_means_classes[cluster].element_count + 1);
                double new_mean = ((double)hueMatrix[i][j]) / (k_means_classes[cluster].element_count + 1);
                // Update k. class properties with new calculating value
                k_means_classes[cluster].cluster_mean = pre_mean + new_mean;
                k_means_classes[cluster].element_count++;
	        }
	    }
	    // Update k_means classes center point
	    for(i = 0; i < k_means_value; i++)
	    {
	        if(k_means_classes[i].element_count != 0)
	        {
	            k_means_classes[i].class_center = k_means_classes[i].cluster_mean;
	        }
	    }

	    // check the classes average point with epsilon value
	    int count = 0;
	    for(i = 0; i < k_means_value; i++)
	    {
	        if(abs(k_means_classes[i].pre_class_center - k_means_classes[i].class_center) < 0.01)
                count++;
	    }
	    // escape the while loop if all means found
	    if(count == k_means_value)
            check = 0;
	}
	// if element count is zero, class center is -360, for easy find that afterwards
	for(i = 0; i < k_means_value; i++)
	{
	    if( k_means_classes[i].element_count == 0)
            k_means_classes[i].class_center = -360;
        printf("\ni: %d Compute cluster %lf elements number %lf",i,k_means_classes[i].class_center,k_means_classes[i].element_count);
	}

    // calculated values write the text
	FILE *f;
	f = fopen("kmeans.txt","w");

    // create the k_means matrix for connecting component labeling
	for(i = 0; i < height; i++)
    {
        for(j = 0; j < width; j++)
        {
            minimum = 360;   // Initialize maximum Hue value
            // Find the nearest pixel
            for(k = 0; k < k_means_value; k++)
            {
                int difference = abs((double)hueMatrix[i][j] - k_means_classes[k].class_center);
                if(difference < minimum)
                {
                    minimum = difference;
                    cluster = k;
                }
            }
            k_meansMatrix[i][j] = cluster + 1;
            fprintf(f, "%d ",k_meansMatrix[i][j]);
        }
        fprintf(f, "%c ",'\n');
    }
    fclose(f);
    return k_meansMatrix;
}

// this section is update labels for connecting component labeling
void update_labeling(int **componentMatrix, int rows, int columns, int width, int new_tag, int update_tag)
{
    int i, j, temp;

    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < width; j++)
        {
            if(!(i == rows && j > columns))
            {
                if(componentMatrix[i][j] == update_tag)
                    componentMatrix[i][j] == new_tag;
            }
        }
    }
}

// section of connecting component labeling for 4-nearest neighbors
int **connecting_component_labeling(int **k_meansMatrix, int height, int width, int k_means_value)
{
    int i, j, k;
    int current_tag;
    int adjacent_row, adjacent_column;
    int tagged, found, ccl_location;
    tag_value = k_means_value + 1;
    int **componentMatrix = (int**)calloc(height, sizeof(int*));
    for(i = 0; i < height; i++)
    {
        componentMatrix[i] = (int*)calloc(width, sizeof(int));
    }
    // looking to neighboring addresses
    //   0 -1   left neigbour
    //  -1 -1   left upper cross neighbor
    //  -1  0   top neighbor
    //  -1  1   right upper cross neighbor
    int directions[4][2] = { {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };

    for(i = 0; i < height; i++)
    {
        for(j = 0; j < width; j++)
        {
            tagged = 0; k = 0; found = 0;
            while(k < 4 && found == 0)
            {
                adjacent_row = i + directions[k][0];
                adjacent_column = j + directions[k][1];
                if(adjacent_row >= 0 && adjacent_row < height && adjacent_column >=0 && adjacent_column < width)
                {
                    if(k_meansMatrix[i][j] == k_meansMatrix[adjacent_row][adjacent_column])
                    {
                        componentMatrix[i][j] =  componentMatrix[adjacent_row][adjacent_column];
                        found = 1;
                    }
                }
                k++;
            }
            if(found == 0)
            {
                componentMatrix[i][j] = tag_value;
                tag_value++;
            }
            else if(i > 0 && j < (width - 1))
            {
                adjacent_row = i + directions[3][0];
                adjacent_column = j + directions[3][1];
                if(k < 3 && k_meansMatrix[i][j] == k_meansMatrix[adjacent_row][adjacent_column] && k_meansMatrix[i][j] != k_meansMatrix[i-1][j])
                {
                    update_labeling(componentMatrix, i, j, width, componentMatrix[i][j], componentMatrix[adjacent_row][adjacent_column]);
                }
            }
        }
    }
    return componentMatrix;
}


int main(int argc, char *argv[])
{
    int height, width, channels, size, k_means_value, hue_size;
    int **hueMatrix;

    unsigned int *hue;
    IplImage* img = 0;
    int i,j,k;
    uchar *data;

    if(argc<2)
    {
        printf("Usage: main <image-file-name>\n\7");
        exit(0);
    }

    // load an image
    img = cvLoadImage(argv[1]);
    if(!img)
    {
        printf("Could not load image file: %s\n",argv[1]);
        exit(0);
    }
    // get the image data
    height    = img->height;
    width     = img->width;
    channels  = img->nChannels;
    data      = (uchar *)img->imageData;

    size = height*width*channels;
    // invert the image
    int rgb[size];
    for(i = 0; i < size; i++)
        rgb[i] = (int)data[i];

    hueMatrix = rgb_to_h(rgb, height, width);

	printf("\n Enter the K cluster number:\n");
    scanf("%d",&k_means_value);
    int **k_meansMatrix = k_means(hueMatrix, height, width, k_means_value);

    printf("\nProcessing a %dx%d image with %d channels\n", height, width, channels);

    // create a window
    cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("mainWin", 100, 100);



    // show the image
    cvShowImage("mainWin", img );

    // wait for a key
    cvWaitKey(0);

    // release the image
    cvReleaseImage(&img );

    return 0;
}

