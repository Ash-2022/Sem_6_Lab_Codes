#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int squareSize = size * size;
    int flatMatrix[squareSize] , rowVector[size];
    int toSearch;
    if (!rank) {
        for (int i = 0; i < squareSize; i++) {
            printf("Enter Element %d : " , i+1);
            fflush(stdout);
            scanf("%d", &flatMatrix[i]);
        }

        printf("Enter Element to Search : ");
        fflush(stdout);
        scanf("%d", &toSearch);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&toSearch , 1 , MPI_INT , 0 , MPI_COMM_WORLD);

    MPI_Scatter(flatMatrix, size , MPI_INT, rowVector, size, MPI_INT, 0, MPI_COMM_WORLD);

    int found = 0;
    for (int i = 0; i < size; i++) {
        if (rowVector[i] == toSearch) {
            found += 1;
        }
    }
    int totalFound = 0;
    MPI_Reduce(&found, &totalFound, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!rank) {
        printf("Element %d is found %d times\n", toSearch, totalFound);
    }
    MPI_Finalize();
    return 0;
}
