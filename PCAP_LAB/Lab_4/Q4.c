#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char inputString[size];
    char toProcess;
    char outputString[size];  

    if (!rank) {
        printf("Enter String of length %d: ", size);
        fflush(stdout);
        scanf("%s", inputString);
    }

    MPI_Barrier(MPI_COMM_WORLD);  
    MPI_Scatter(inputString, 1, MPI_CHAR, &toProcess, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    
    memset(outputString, '\0', size);  
    for (int i = 0; i <= rank; i++) {
        outputString[i] = toProcess;
    }
    outputString[size] = '\0';

    printf("Process %d created string: %s\n", rank, outputString);
    fflush(stdout);  

    char resultString[size * size];  
    memset(resultString, '\0', sizeof(resultString));  

    MPI_Gather(outputString, size, MPI_CHAR, resultString, size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (!rank) {
        printf("Final gathered string: ");
        for (int i = 0; i < size * size; i++) {
            if (resultString[i] != '\0') {
                printf("%c", resultString[i]);
            }
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
