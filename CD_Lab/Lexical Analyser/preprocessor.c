#include "preprocessor.h"

int main() 
{ 
    FILE* input = fopen("bbc.c", "r");
    FILE* output = fopen("preprocessed.c", "w");
    if (input && output) {
        preprocess(input, output);
        fclose(input);
        fclose(output);
    }
    return 0;  
} 
