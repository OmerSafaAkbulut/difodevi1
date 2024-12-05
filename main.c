#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define SAMPLE_SIZE 100
#define N 28
#define SPLIT_RATIO 0.8
#define LEARNING_RATE 0.0001
#define ITERATIONS 30
#define INPUT_FILE "test_in_merged.txt"

// Resim datalarını tutan struct
typedef struct {
    double pixels[N * N];
    int class;
} ImageData;

void print_dataset(ImageData*, int);
void shuffle_dataset(ImageData*, int);
void split_dataset(ImageData*, ImageData*, ImageData*, int, int);
void print_image(ImageData);
void gradient_descent(ImageData*, double*, int, int);
void sgd(ImageData*, double*, int, int);
void adam(ImageData*, double*, int, int, double*, double*, int);
void initialize_weights(double**, int);
double calculate_gradient(double, double, double);
double calculate_accuracy(ImageData*, double*, int, int);
double calculateMSE(ImageData*, double*, int, int);
double calculate(double*, double*, int);

// Creates random datasets everytime
void generate_dataset(ImageData* dataset, int total_samples) {
    // Open Input File
    FILE* file_in = fopen(INPUT_FILE, "r");
    if (file_in == NULL) { printf("Couldn't open file\n"); return; }

    for (int i = 0; i < total_samples; i++) {
        fscanf(file_in, "%d,", &dataset[i].class); // Read Label
        for (int j = 0; j < N * N; j++) {
            // Read Pixels and Normalize
            fscanf(file_in, "%lf,", &dataset[i].pixels[j]);
            dataset[i].pixels[j] /= 255.0;
        }
    }
    /* // Iterate each sample
    for (int i = 0; i < total_samples; i++) {
        dataset[i].class = (i < total_samples / 2) ? 1 : -1;
        // Iterate each row
        for (int j = 0; j < N; j++) {
            // Iterate each coloumn
            for (int k = 0; k < N; k++) {
                double pixel;
                // We are creating a gradient image as class 1
                // Range is between 100,255
                // 0 to 50 added by random
                // Rest is defined by height of the pixels
                if(dataset[i].class == 1) pixel = (double)(100 +(rand()%51) + (j*105)/(N-1));

                else {
                    // We are creating a checkered image as class 1
                    // Range is between 105,255
                    // 0 to 50 added by random
                    int square_size = N/5;
                    if (((j / square_size) + (k / square_size)) % 2 == 0) {
                        pixel = 55 + (rand()%51);  // Dark square
                    }
                    else {
                        pixel = 205 + (rand()%51);  // Light square
                    }
                }
                // Divided by 255 to normalize
                dataset[i].pixels[j*N+k] = pixel / 255.0;
            }
        }
    }*/
}

int main() {
    srand(time(NULL));

    // Clock variables
    clock_t start_t, end_t;
    double total_t;

    // Calculate the const variables
    const int total_samples = 2 * SAMPLE_SIZE;
    const int train_set_size = total_samples * SPLIT_RATIO;
    const int test_set_size = total_samples - train_set_size;
    const int input_size = N * N;

    // Allocate datasets
    ImageData* dataset = (ImageData*)malloc(sizeof(ImageData) * total_samples);
    ImageData* train_set = (ImageData*)malloc(sizeof(ImageData) * train_set_size);
    ImageData* test_set = (ImageData*)malloc(sizeof(ImageData) * test_set_size);

    generate_dataset(dataset, total_samples);

    // Split the dataset to subsets according to SPLIT_RATIO
    split_dataset(dataset, train_set, test_set, train_set_size, test_set_size);

    /*printf("Eğitim kümesi:\n");
    print_dataset(train_set, train_set_size);

    printf("\nTest kümesi:\n");
    print_dataset(test_set, test_set_size);*/

    // Allocate ADAM variables
    double* m = (double*)calloc(input_size, sizeof(double));
    double* v = (double*)calloc(input_size, sizeof(double));

    // Define iteration count for each method
    int iterations = ITERATIONS;

    // Modes 1=gd 2=sgd 3=adam
    for (int mode = 1; mode <= 3; mode++)
    {
        double* weights;
        initialize_weights(&weights, input_size);

        char name[20];
        char tempName[20];

        // Get current method name
        switch (mode)
        {
        case 1:
            printf("Gradient Descent:\n");
            strcpy(name, "gd");
            break;
        case 2:
            printf("\nSGD:\n");
            strcpy(name, "sgd");
            break;
        case 3:
            printf("\nAdam:\n");
            strcpy(name, "adam");
            break;
        }

        // FILE OPERATIONS
        sprintf(tempName, "%s_stats.csv", name);
        FILE* statsFile = fopen(tempName, "w");

        sprintf(tempName, "%s_weights.csv", name);
        FILE* weightsFile = fopen(tempName, "w");

        if (weightsFile == NULL || statsFile == NULL) { printf("Couldn't open file\n"); return 1; }
        fprintf(statsFile, "iter,train_mse,test_mse,train_acc,test_acc,time\n");
        for (int i = 0; i < input_size; ++i) {
            fprintf(weightsFile, "w%d", i);
            if (i < input_size - 1) fprintf(weightsFile, ",");
            else fprintf(weightsFile, "\n");
        }

        // Cache Clock
        start_t = clock();

        for (int i = 0; i < iterations; i++) {
            // Execute current method
            switch (mode)
            {
            case 1:
                gradient_descent(train_set, weights, train_set_size, input_size);
                break;
            case 2:
                for (int j = 0; j < train_set_size; j++) sgd(train_set, weights, train_set_size, input_size);
                break;
            case 3:
                for (int t = 1; t <= train_set_size; t++) adam(train_set, weights, train_set_size, input_size, m, v, t);
                break;
            }

            end_t = clock(); // Get clock
            total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC; // Calculate time

            // Get training statistics
            double train_MSE = calculateMSE(train_set, weights, train_set_size, input_size);
            double train_accuracy = calculate_accuracy(train_set, weights, train_set_size, input_size);

            // Get testing statistics
            double test_MSE = calculateMSE(test_set, weights, test_set_size, input_size);
            double test_accuracy = calculate_accuracy(test_set, weights, test_set_size, input_size);

            // Print info to console
            printf("Iteration %d: Train MSE = %.4f, Train Accuracy = %%%.2f, Test MSE = %.4f, Test Accuracy = %%%.2f, Time = %f\n", i + 1,
                train_MSE, train_accuracy, test_MSE, test_accuracy, total_t);

            // Write statistics
            fprintf(statsFile, "%d,%.4f,%.4f,%.2f,%.2f,%f\n", i + 1, train_MSE, test_MSE, train_accuracy, test_accuracy, total_t);
            // Write weights
            
            for (int j = 0; j < input_size; j++) {
                fprintf(weightsFile, "%.8f", weights[j]);
                if (j < input_size - 1) fprintf(weightsFile, ",");
                else fprintf(weightsFile, "\n");
            }
        }

        fclose(statsFile);
        fclose(weightsFile);
        free(weights);
    }

    // Free memories
    free(m);
    free(v);
    free(dataset);
    free(test_set);
    free(train_set);

    return 0;
}

// Initializes weights by random decimals
void initialize_weights(double** weights, int size) {
    (*weights) = (double*)malloc(size * sizeof(double)); // Allocate memory

    for (int i = 0; i < size; i++) {
        //(*weights)[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Normalizes to [0, 1] then [-1, 1]
        (*weights)[i] = 0.01;
    }
}

// Function that performs gradient descent. Every sample is calculated every iteration (w_new = w_old - learningRate * J(w_i))
void gradient_descent(ImageData* dataset, double* weights, int sample_size, int input_size) {
    // Initializing the gradient array
    double* gradients = (double*)calloc(input_size, sizeof(double));

    // Filling up the gradient array
    for (int i = 0; i < sample_size; i++) {
        double output = calculate(weights, dataset[i].pixels, input_size);
        double error = dataset[i].class - output;

        for (int j = 0; j < input_size; j++) {
            gradients[j] += calculate_gradient(output, error, dataset[i].pixels[j]);
        }
    }

    // Updating the weights
    for (int i = 0; i < input_size; i++) {
        weights[i] -= LEARNING_RATE * (gradients[i] / sample_size);
    }

    free(gradients);
}

// Function that performs sgd. Only one sample is calculated every iteration (w_new = w_old - learningRate * J(w_i))
void sgd(ImageData* dataset, double* weights, int sample_size, int input_size) {
    // Selecting random sample
    int sample_index = rand() % sample_size;

    double output = calculate(weights, dataset[sample_index].pixels, input_size);
    double error = dataset[sample_index].class - output;

    // Updating the weights
    for (int i = 0; i < input_size; i++) {
        weights[i] -= LEARNING_RATE * calculate_gradient(output, error, dataset[sample_index].pixels[i]);
    }
}

// Function that performs adam. Functions are according to class documents
void adam(ImageData* dataset, double* weights, int sample_size, int input_size, double* m, double* v, int t) {
    // Default variables according to documents
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    int sample_index = (rand() % sample_size);
    double* gradients = (double*)calloc(input_size, sizeof(double));
    double output = calculate(weights, dataset[sample_index].pixels, input_size);
    double error = dataset[sample_index].class - output;

    for (int i = 0; i < input_size; i++) {
        gradients[i] = calculate_gradient(output, error, dataset[sample_index].pixels[i]);
    }

    for (int i = 0; i < input_size; i++) {
        // Calculating variables
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];

        double m_hat = m[i] / (1 - pow(beta1, t));
        double v_hat = v[i] / (1 - pow(beta2, t));

        // Updating the weights
        weights[i] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + epsilon);
    }

    free(gradients);
}

// Calculates the accuracy of the model
double calculate_accuracy(ImageData* dataset, double* weights, int sample_size, int input_size) {
    int correct = 0;
    for (int i = 0; i < sample_size; i++) {
        double output = calculate(weights, dataset[i].pixels, input_size);
        int prediction = ((output > 0) ? 1 : -1); // Round the prediction
        if (prediction == dataset[i].class) correct++; // Increment if correct
    }
    return (double)correct / sample_size * 100.0; // Turn to percentage
}

// Calculates Mean Square Error for the dataset
double calculateMSE(ImageData* dataset, double* weights, int sample_size, int input_size) {
    // MSE = 1/n Sum((Y - Y_out)^2)
    double mse = 0.0;

    for (int i = 0; i < sample_size; i++) {
        double output = calculate(weights, dataset[i].pixels, input_size); // Y_out
        double error = dataset[i].class - output; // Y - Y_out
        mse += error * error; // (Y - Y_out)^2
    }

    return (mse / sample_size); // Sum divided by n
}

// Calculates output for an image
double calculate(double* weights, double* inputs, int size) {
    double temp = 0.0;
    for (int i = 0; i < size; i++)
    {
        // Dot product
        temp += weights[i] * inputs[i];
    }
    return tanh(temp);
}

// Calculates gradient
double calculate_gradient(double output, double error, double x) {
    // Hata fonksiyonun turevi alinir (y - y_out)^2 -> -2*(y - y_out)y_hat -> -2*error*(1 - y_out^2)*x
    return (-2 * error * (1 - output * output) * x);
}

// Shuffles the dataset for better splitting
void shuffle_dataset(ImageData* dataset, int size) {
    // Common shuffling algorithm
    for (int i = (size - 1); i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap (don't swap if same index)
        if (i != j) {
            ImageData temp = dataset[i];
            dataset[i] = dataset[j];
            dataset[j] = temp;
        }
    }
}

// Splits dataset into train and test sets
void split_dataset(ImageData* dataset, ImageData* train_set, ImageData* test_set, int train_set_size, int test_set_size) {
    //shuffle_dataset(dataset, train_set_size + test_set_size);
    // Train Set
    for (int i = 0; i < train_set_size; i++) {
        train_set[i] = dataset[i];
    }

    // Test Set
    for (int i = 0; i < test_set_size; i++) {
        test_set[i] = dataset[train_set_size + i];
    }
}

// Prints array of images
void print_dataset(ImageData* dataset, int count) {
    for (int i = 0; i < count; i++)
    {
        printf("Image: %d ", i + 1);
        print_image(dataset[i]);
    }

}

// Prints info about image
void print_image(ImageData img) {
    printf("Class: %d\n", img.class);
    for (int i = 0; i < N * N; i++)
    {
        // Print normalized pixels
        printf("%.2f ", img.pixels[i]);

        // End line if its end of the row
        if (((i + 1) % N) == 0) printf("\n");
    }
    printf("\n");
}
