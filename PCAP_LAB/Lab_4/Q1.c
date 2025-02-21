#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_N 100

int factorial(int n) {
    if (n < 0) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_code = MPI_ERR_OTHER;
        int length_of_error_string;

        MPI_Error_string(error_code, error_string, &length_of_error_string);

        fprintf(stderr, "Error in factorial: Factorial of a negative number (%d) is undefined.\n", n);
        fprintf(stderr, "MPI Error: %s\n", error_string);

        MPI_Abort(MPI_COMM_WORLD, error_code); 
        return -1; 
    }

    if (n == 0 || n == 1) return 1;
    return n * factorial(n - 1);
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[MAX_N], recvNum;

    if (rank == 0) {
        for (int i = 1; i <= size; i++) {
            sendbuf[i - 1] = i;
        }
    }

    MPI_Scatter(sendbuf, 1, MPI_INT, &recvNum, 1, MPI_INT, 0, MPI_COMM_WORLD);

    recvNum = factorial(recvNum);

    printf("Factorial Result: Value in rank %d is %d\n", rank, recvNum);

    MPI_Scan(&recvNum, sendbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("Prefix Sum Result: Value in rank %d is %d\n", rank, sendbuf[0]);

    MPI_Finalize();
    return 0;
}
