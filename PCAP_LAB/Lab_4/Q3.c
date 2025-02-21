#include <mpi.h>
#include <stdio.h>

void transposeMatrix(int *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            int temp = matrix[i * size + j];
            matrix[i * size + j] = matrix[j * size + i];
            matrix[j * size + i] = temp;
        }
    }
}


int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int squareSize = size * size;
    int flatMatrix[squareSize] , rowVector[size];

    if (!rank) {
        for (int i = 0; i < squareSize; i++) {
            printf("Enter Element %d : " , i+1);
            fflush(stdout);
            scanf("%d", &flatMatrix[i]);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&flatMatrix , squareSize , MPI_INT , 0 , MPI_COMM_WORLD);
    
    for(int i = 0; i < size; i++){
        rowVector[i] = flatMatrix[i * size + rank];
    }

    for(int i = 1; i < size; i++){
        rowVector[i] += rowVector[i-1]; 
    }

    MPI_Gather(rowVector, size , MPI_INT, flatMatrix, size, MPI_INT, 0, MPI_COMM_WORLD);

    if (!rank) {
        transposeMatrix(flatMatrix , size);
        for (int i = 0; i < squareSize; i++) {
            printf("%d ", flatMatrix[i]);
            if((i + 1) % size == 0) printf("\n");
        }
    }
    MPI_Finalize();
    return 0;
}
