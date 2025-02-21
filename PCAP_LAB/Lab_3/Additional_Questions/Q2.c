#include <mpi.h>
#include <stdio.h>

int main(int argc , char*argv[]){
    int rank , size;
    MPI_Init(&argc , &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &size);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    int input[size];
    int evenCount = 0 , oddCount = 0;
    if(!rank){
        for(int i = 0; i < size;i++){
            printf("Enter Element %d : " , i);
            fflush(stdout);
            scanf("%d" , &input[i]);
        }
    }
    MPI_Bcast(&input , size , MPI_INT , 0 , MPI_COMM_WORLD);
    int correctedNum = 0;
    if(input[rank] % 2){
        correctedNum = 0;
        oddCount = 1;
    } 
    else correctedNum = 1;
    MPI_Gather(&correctedNum , 1 , MPI_INT , &input , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Reduce(&correctedNum , &evenCount , 1 , MPI_INT , MPI_SUM , 0 , MPI_COMM_WORLD);
    if(!rank){
        for(int i = 0; i < size;i++){
            printf("%d " , input[i]);
            fflush(stdout);
        }
        printf("\n");
        printf("Even Count = %d\nOdd Count = %d\n", evenCount , (size - evenCount));
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
