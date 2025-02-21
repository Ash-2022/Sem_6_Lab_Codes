#include <mpi.h>
#include <stdio.h>

int reverseNumber(int num){
    int reversedNum = 0;
    while(num){
        // printf("Num = %d\n" , reversedNum);
        reversedNum *= 10;
        reversedNum += num % 10;
        num /= 10;
    }
    return reversedNum;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int inputNum[9] = {18 , 523 , 301 , 1234 , 2 , 14 , 108 , 150 , 1928};
    for(int i = 0; i < 9; i ++){
        if(rank == i){
            inputNum[i] = reverseNumber(inputNum[i]);
        }
    }
    printf(" %d " , inputNum[rank]);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
