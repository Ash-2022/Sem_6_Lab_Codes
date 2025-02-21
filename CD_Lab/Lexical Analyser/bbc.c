/*
    SYSTEM HEADERS
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef WIN64
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

#define VERSION "1.1"

/*
    MACROS
*/

// Defining the BitBoard Datatype
#define U64 unsigned long long

// Defining Set , Get , Pop(Remove) Bit Macros
#define setBitOnSquare(bitBoard , square) (bitBoard |= (1ULL << square))
#define getBitOnSquare(bitBoard , square) (bitBoard & (1ULL << square))
#define popBitOnSquare(bitBoard , square) (bitBoard &= (~(1ULL << square)))

/*
    BIT MANIPULATIONS TRICKS
*/

static inline int countBits(U64 bitBoard){
    // Bit Count
    int count = 0;
    // Consecutively reset LS1B
    while(bitBoard){
        //increment count
        count++;
        //Reset LS1B
        bitBoard &= (bitBoard - 1);
    }

    return count;

}

static inline int getLS1BIndex(U64 bitBoard){
    if (bitBoard) return countBits((bitBoard & -bitBoard) - 1);
    return -1;
}

/*
    ENUMS FOR EASE OF USE
*/

// Board Squares
enum{
    a8 , b8 ,c8 ,d8 ,e8 ,f8 ,g8 ,h8,
    a7 , b7 ,c7 ,d7 ,e7 ,f7 ,g7 ,h7,
    a6 , b6 ,c6 ,d6 ,e6 ,f6 ,g6 ,h6,
    a5 , b5 ,c5 ,d5 ,e5 ,f5 ,g5 ,h5,
    a4 , b4 ,c4 ,d4 ,e4 ,f4 ,g4 ,h4,
    a3 , b3 ,c3 ,d3 ,e3 ,f3 ,g3 ,h3,
    a2 , b2 ,c2 ,d2 ,e2 ,f2 ,g2 ,h2,
    a1 , b1 ,c1 ,d1 ,e1 ,f1 ,g1 ,h1 , NULL_SQUARE
};

// NOTE :  << goes Right ,i.e, opposite to arrow
// >> goes down and vice-versa after boundary

// Board Colors
enum{
    WHITE , BLACK , BOTH
};

// bishop and rook
enum { rook, bishop };

// Castling binary representations
enum{
    WK = 1 , WQ = 2 , BK = 4 , BQ = 8
};

// Piece binary Representations
enum {
    P , N , B , R , Q , K , 
    p , n , b , r , q , k
};

// Differentiate moves based on Quite and Capture moves --> For Copy/Make and Quiesence Search
enum {allMoves , captureMoves};

/*
    FILE MASKS -> Number to AND with BitBoard so that the calculations do not use that file -> set to 0 only for that file(s)
*/

const U64 maskFileA = 18374403900871474942ULL;
const U64 maskFileAB = 18229723555195321596ULL;
const U64 maskFileH = 9187201950435737471ULL;
const U64 maskFileGH = 4557430888798830399ULL;

const char* squareToCoordinates[] = {
    "a8" , "b8" ,"c8" ,"d8" ,"e8" ,"f8" ,"g8" ,"h8",
    "a7" , "b7" ,"c7" ,"d7" ,"e7" ,"f7" ,"g7" ,"h7",
    "a6" , "b6" ,"c6" ,"d6" ,"e6" ,"f6" ,"g6" ,"h6",
    "a5" , "b5" ,"c5" ,"d5" ,"e5" ,"f5" ,"g5" ,"h5",
    "a4" , "b4" ,"c4" ,"d4" ,"e4" ,"f4" ,"g4" ,"h4",
    "a3" , "b3" ,"c3" ,"d3" ,"e3" ,"f3" ,"g3" ,"h3",
    "a2" , "b2" ,"c2" ,"d2" ,"e2" ,"f2" ,"g2" ,"h2",
    "a1" , "b1" ,"c1" ,"d1" ,"e1" ,"f1" ,"g1" ,"h1"
};

/*
    BOARD REPRESENTATION
            ALL TOGETHER

    8  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
    7  ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎
    6  .  .  .  .  .  .  .  .
    5  .  .  .  .  .  .  .  .
    4  .  .  .  .  .  .  .  .
    3  .  .  .  .  .  .  .  .
    2  ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
    1  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖

        a b c d e f g h

    CASTLING REPRESENTATION

    bin  dec
    
   0001    1  white king can castle to the king side
   0010    2  white king can castle to the queen side
   0100    4  black king can castle to the king side
   1000    8  black king can castle to the queen side

   examples

   1111       both sides an castle both directions
   1001       black king => queen side
              white king => king side


*/

// ASCII Piece Representation for CLI
char asciiPieces[] = "PNBRQKpnbrqk";

// UNICODE(These are strings not char use %s) Piece Representation for CLI
char *unicodePieces[12] =  {
    "♙" , "♘" , "♗" , "♖" , "♕" , "♔" , 
    "♟︎" , "♞" , "♝" , "♜" , "♛" , "♚" ,
};

int asciiToIntegerMapping[] = {
    ['P'] = P,
    ['N'] = N,
    ['B'] = B,
    ['R'] = R,
    ['Q'] = Q,
    ['K'] = K,
    ['p'] = p,
    ['n'] = n,
    ['b'] = b,
    ['r'] = r,
    ['q'] = q,
    ['k'] = k,
};

char promotedPieces[] = {
    [Q] = 'q',
    [R] = 'r',
    [B] = 'b',
    [N] = 'n',
    [q] = 'q',
    [r] = 'r',
    [b] = 'b',
    [n] = 'n',
};

// Define all piece BitBoard -> 1 for every piece of each color => 6 * 2 = 12
U64 pieceBitBoards[12];

// define occupancy bitBoard -> 1 for WHITE , BLACK , and whole board => 3
U64 boardOccupancies[3];

// Tracks which side's tur it is to move
int sideToMove;

// Tracks the EnPassant Square
int enPassant = NULL_SQUARE;

// Tracks the Castling Rights
int castlingRights;

// Position Repetition Table 
U64 repetitionTable[1000]; // 1000 ply => 500 moves , whichi is a very long game 

// Repetition Index
int repetitionIndex;

//Half Move Counter 
int ply; // Sum of moves made by both sides from the current position
//eg : if 3 move by white and 2 by black are made during the search => 5(=3+2)ply from current position

/*
    XORSHIFT32 PSEUDO RANDOM NUMBER GENERATOR
*/

unsigned int seed = 1804289383;

unsigned int getRandomU32Number() {
    unsigned int number = seed;
    number ^= number << 13;
    number ^= number >> 17;
    number ^= number << 5;
    seed = number;
    return number;
}


U64 getRandomU64Number(){
    U64 n1 , n2 , n3 , n4;
    n1 = (U64) (getRandomU32Number()) & 0xFFFF;
    n2 = (U64) (getRandomU32Number()) & 0xFFFF;
    n3 = (U64) (getRandomU32Number()) & 0xFFFF;
    n4 = (U64) (getRandomU32Number()) & 0xFFFF;

    return n1 | (n2 << 16) | (n3 << 32) | (n4 << 48);
}

/*
    ZOBRIST HASHING -> Maps positions to specific numbers (fixed size length)
    Order to hash : 
    1) Piece Position/Placement
    2) Enpassant Possibility
    3) Castling Possibility
    4) Side To move
*/

// Creating a position hash key variable to use in hashing
U64 hashKey;

// Random Piece Key [piece][square]
U64 pieceKeys[12][64];

// Random EnPassank Keys[square]
U64 enPassantKeys[64];

// Random Castling Keys
U64 castlingKeys[16]; // Max = 1111 => 15 in decimal

// Random Side to Move Keys , consider for hashing only if side to move = BLACK
U64 sideToMoveKey;

// Init Random Hash Keys
void initRandomKeys(){

    // Since using random numbers , we use the seed from the magic number generator
    // seed = 1804289383 for reference

    // Loop Over all possible pieces
    for(int piece = P; piece <= k; piece++){
        // Loop over all square
        for(int square = 0; square < 64; square++){
            
            // Init Random Piece keys
            pieceKeys[piece][square] = getRandomU64Number();
        
        }
    }
    
    for(int square = 0; square < 64; square++){
        
        // Init Random EnPassant keys
        enPassantKeys[square] = getRandomU64Number();
    
    }

    // Loop over the castling keys
    for(int idx = 0; idx < 16; idx ++){

        // Init Random Castling keys
        castlingKeys[idx] = getRandomU64Number();

    }

    // Init Random sideToMove keys
    sideToMoveKey = getRandomU64Number();

}

U64 generateZobristHashKeys(){

    //Final hash Key for the Position
    U64 positionKey = 0ULL;

    // temporary piece's BitBoard holder to check which piece we are generating hash for
    U64 copyBitBoard;

    // Loop over piece boards
    for(int piece = P; piece <= k; piece++){
        
        // Init piece's BitBoard copy
        copyBitBoard = pieceBitBoards[piece];

        // Loop over the bit board till there are no pieces of that type
        while(copyBitBoard){
            // Init square occupied by the piece
            int square = getLS1BIndex(copyBitBoard);

            // Hash the piece with different hash for different position of the pieces
            positionKey ^= pieceKeys[piece][square];

            // Pop the LS1B
            popBitOnSquare(copyBitBoard , square);

        }

    }


    if (enPassant != NULL_SQUARE){

        // Combine the enpassant hash into the position key
        positionKey ^= enPassantKeys[enPassant];
    }

    // Combine the castling hash into the position key
    positionKey ^= castlingKeys[castlingRights];

    // Combine the side to move into the hash key , hash only if side == BLACK => reduce computation
    if (sideToMove == BLACK) positionKey ^= sideToMoveKey;

    // Return the hash key
    return positionKey;
}

/*
    I/O
*/

// Print Bitboard Function
void printBitBoard(U64 bitBoard){
    printf("\n");
    // loop over the board ranks 
    for(int rank = 0; rank < 8 ;rank++){
        // Loop over the board files 
        for(int file = 0; file < 8; file++){
            // Convert rank,file into a square index
            int square = rank * 8 + file;
            // Printing ranks , 
            if (!file){
                // !file same as file == 0;
                //(since 0 is false and other numbers are true in C/C++)

                printf("%d " , 8-rank);
            }
            // Printing the bit state(0 or 1) for that square index
            printf(" %d" , getBitOnSquare(bitBoard , square) ? 1 : 0);
        }
        printf("\n");
    }
    // Printing the board ranks
    printf("\n   a b c d e f g h \n");

    // Printing the bitboard in unsigned decimal value
    printf("\n   Bitboard = %llud \n" , bitBoard);

}

void printBoard(){
    // Print Offset
    printf("\n");

    for(int rank = 0 ; rank < 8;  rank++){
        for(int file = 0; file < 8; file++){
            // Generate square from rank and file
            int square = 8 * rank + file;
            
            // Generate rank numbers for the board display
            if(!file) printf("%d " , (8 - rank));

            // -1 =>Empty/No Piece , else the piece enum value will be loaded into it.
            int piece = -1;

            // Loop over all Piece BitBoards
            for(int consideredPiece = P; consideredPiece <=k ; consideredPiece++){
                if(getBitOnSquare(pieceBitBoards[consideredPiece] , square)) piece = consideredPiece;
            }
            // If no Piece print " . " , else print the ascii/unicode (based on OS) of the piece referenced by the index.
            #ifdef WIN_64
                printf(" %c " , ((piece == -1) ? ' .' : asciiPieces[piece]));
            #else 
                printf(" %s" , ((piece == -1) ? "." : unicodePieces[piece]));
            #endif
        }
        printf("\n");
    }
    // Generate file numbers for the board display
    printf("\n   a b c d e f g h \n");
    printf("   Side to move : %s\n" , ((!sideToMove) ? "WHITE" : "BLACK"));
    printf("   EnPassant : %s\n" , ((enPassant != NULL_SQUARE) ? squareToCoordinates[enPassant] : "NO"));
    printf("   Castling Rights : %c%c%c%c\n",
        (castlingRights & WK) ? 'K' : '-',
        (castlingRights & WQ) ? 'Q' : '-',
        (castlingRights & BK) ? 'k' : '-',
        (castlingRights & BQ) ? 'q' : '-'
    );
    printf("   Position Hash : %llx\n" , hashKey);
}

/*
    OCCUPANCY MASKS
*/

const int bishopOccupancyBits[64] = {
    6 ,  5 ,  5 ,  5 ,  5 ,  5 ,  5 ,  6 , 
    5 ,  5 ,  5 ,  5 ,  5 ,  5 ,  5 ,  5 , 
    5 ,  5 ,  7 ,  7 ,  7 ,  7 ,  5 ,  5 , 
    5 ,  5 ,  7 ,  9 ,  9 ,  7 ,  5 ,  5 , 
    5 ,  5 ,  7 ,  9 ,  9 ,  7 ,  5 ,  5 , 
    5 ,  5 ,  7 ,  7 ,  7 ,  7 ,  5 ,  5 , 
    5 ,  5 ,  5 ,  5 ,  5 ,  5 ,  5 ,  5 , 
    6 ,  5 ,  5 ,  5 ,  5 ,  5 ,  5 ,  6 
};

const int rookOccupancyBits[64] = {
    12 ,  11 ,  11 ,  11 ,  11 ,  11 ,  11 ,  12 , 
    11 ,  10 ,  10 ,  10 ,  10 ,  10 ,  10 ,  11 , 
    11 ,  10 ,  10 ,  10 ,  10 ,  10 ,  10 ,  11 , 
    11 ,  10 ,  10 ,  10 ,  10 ,  10 ,  10 ,  11 , 
    11 ,  10 ,  10 ,  10 ,  10 ,  10 ,  10 ,  11 , 
    11 ,  10 ,  10 ,  10 ,  10 ,  10 ,  10 ,  11 , 
    11 ,  10 ,  10 ,  10 ,  10 ,  10 ,  10 ,  11 , 
    12 ,  11 ,  11 ,  11 ,  11 ,  11 ,  11 ,  12
};

/*
    MAGIC NUMBERS -> Constants for a square
*/

U64 rookMagicNumbers[64] = {
    0x8a80104000800020ULL ,
    0x140002000100040ULL ,
    0x2801880a0017001ULL ,
    0x100081001000420ULL ,
    0x200020010080420ULL ,
    0x3001c0002010008ULL ,
    0x8480008002000100ULL ,
    0x2080088004402900ULL ,
    0x800098204000ULL ,
    0x2024401000200040ULL ,
    0x100802000801000ULL ,
    0x120800800801000ULL ,
    0x208808088000400ULL ,
    0x2802200800400ULL ,
    0x2200800100020080ULL ,
    0x801000060821100ULL ,
    0x80044006422000ULL ,
    0x100808020004000ULL ,
    0x12108a0010204200ULL ,
    0x140848010000802ULL ,
    0x481828014002800ULL ,
    0x8094004002004100ULL ,
    0x4010040010010802ULL ,
    0x20008806104ULL ,
    0x100400080208000ULL ,
    0x2040002120081000ULL ,
    0x21200680100081ULL ,
    0x20100080080080ULL ,
    0x2000a00200410ULL ,
    0x20080800400ULL ,
    0x80088400100102ULL ,
    0x80004600042881ULL ,
    0x4040008040800020ULL ,
    0x440003000200801ULL ,
    0x4200011004500ULL ,
    0x188020010100100ULL ,
    0x14800401802800ULL ,
    0x2080040080800200ULL ,
    0x124080204001001ULL ,
    0x200046502000484ULL ,
    0x480400080088020ULL ,
    0x1000422010034000ULL ,
    0x30200100110040ULL ,
    0x100021010009ULL ,
    0x2002080100110004ULL ,
    0x202008004008002ULL ,
    0x20020004010100ULL ,
    0x2048440040820001ULL ,
    0x101002200408200ULL ,
    0x40802000401080ULL ,
    0x4008142004410100ULL ,
    0x2060820c0120200ULL ,
    0x1001004080100ULL ,
    0x20c020080040080ULL ,
    0x2935610830022400ULL ,
    0x44440041009200ULL ,
    0x280001040802101ULL ,
    0x2100190040002085ULL ,
    0x80c0084100102001ULL ,
    0x4024081001000421ULL ,
    0x20030a0244872ULL ,
    0x12001008414402ULL ,
    0x2006104900a0804ULL ,
    0x1004081002402ULL 
};
U64 bishopMagicNumbers[64] = {
    0x40040822862081ULL ,
    0x40810a4108000ULL ,
    0x2008008400920040ULL ,
    0x61050104000008ULL ,
    0x8282021010016100ULL ,
    0x41008210400a0001ULL ,
    0x3004202104050c0ULL ,
    0x22010108410402ULL ,
    0x60400862888605ULL ,
    0x6311401040228ULL ,
    0x80801082000ULL ,
    0x802a082080240100ULL ,
    0x1860061210016800ULL ,
    0x401016010a810ULL ,
    0x1000060545201005ULL ,
    0x21000c2098280819ULL ,
    0x2020004242020200ULL ,
    0x4102100490040101ULL ,
    0x114012208001500ULL ,
    0x108000682004460ULL ,
    0x7809000490401000ULL ,
    0x420b001601052912ULL ,
    0x408c8206100300ULL ,
    0x2231001041180110ULL ,
    0x8010102008a02100ULL ,
    0x204201004080084ULL ,
    0x410500058008811ULL ,
    0x480a040008010820ULL ,
    0x2194082044002002ULL ,
    0x2008a20001004200ULL ,
    0x40908041041004ULL ,
    0x881002200540404ULL ,
    0x4001082002082101ULL ,
    0x8110408880880ULL ,
    0x8000404040080200ULL ,
    0x200020082180080ULL ,
    0x1184440400114100ULL ,
    0xc220008020110412ULL ,
    0x4088084040090100ULL ,
    0x8822104100121080ULL ,
    0x100111884008200aULL ,
    0x2844040288820200ULL ,
    0x90901088003010ULL ,
    0x1000a218000400ULL ,
    0x1102010420204ULL ,
    0x8414a3483000200ULL ,
    0x6410849901420400ULL ,
    0x201080200901040ULL ,
    0x204880808050002ULL ,
    0x1001008201210000ULL ,
    0x16a6300a890040aULL ,
    0x8049000441108600ULL ,
    0x2212002060410044ULL ,
    0x100086308020020ULL ,
    0x484241408020421ULL ,
    0x105084028429c085ULL ,
    0x4282480801080cULL ,
    0x81c098488088240ULL ,
    0x1400000090480820ULL ,
    0x4444000030208810ULL ,
    0x1020142010820200ULL ,
    0x2234802004018200ULL ,
    0xc2040450820a00ULL ,
    0x2101021090020ULL 
};

/*
    FEN-> BOARD
*/

// FEN Debug Positions

#define EMPTY_BOARD "8/8/8/8/8/8/8/8 w - - "
// 8 same as 11111111 in FEN 
#define START_POSITION "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
#define TRICKY_POSITION "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 "
#define KILLER_POSITION "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1"
#define CMK_POSITION "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9 "
#define REPETITION_POSITION "2r3k1/R7/8/1R6/8/8/P4KPP/8 w - - 0 40 "

void parseFENString(char* fen){
    
    // Reset Board Position and Variables
    memset(pieceBitBoards , 0ULL , sizeof(pieceBitBoards));
    memset(boardOccupancies , 0ULL , sizeof(boardOccupancies));
    sideToMove = 0;
    enPassant = NULL_SQUARE;
    castlingRights = 0;

    // Reset Repetition Index
    repetitionIndex = 0;

    // Reset Repetition Table
    memset(repetitionTable , 0 , sizeof(repetitionTable));

    // Loop over all files and ranks
    for(int rank = 0; rank < 8; rank++){

        for(int file = 0; file < 8; file++){
            // Init Square using file and rank
            int square = 8 * rank + file;

            // match the Ascii from FEN with the pieces     
            if (((*fen >= 'a') && (*fen <= 'z')) || ((*fen >= 'A') && (*fen <= 'Z'))){
                // Init Piece Type
                int piece = asciiToIntegerMapping[*fen];

                // Set piece in corresponding bitBoard
                setBitOnSquare(pieceBitBoards[piece], square);

                // Increment pointer of the FEN string to get the next char of FEN
                fen++;
            }

            // Map 1 from FEN to empty sqaure
            if ((*fen >= '0') && (*fen <='9')){
                // Init Offset for Number in String to Integer
                int offset = *fen - '0';

                // -1 =>Empty/No Piece , else the piece enum value will be loaded into it.
                int piece = -1;

                // Loop over all Piece BitBoards
                for(int consideredPiece = P; consideredPiece <=k ; consideredPiece++){
                    // Map Piece if present on current sqaure
                    if(getBitOnSquare(pieceBitBoards[consideredPiece] , square)) piece = consideredPiece;
                }

                // On Empty Square
                if (piece == -1) file--;

                // Adjust File Counter
                file += offset;

                // Increment the FEN String Pointer
                fen++;
            }

            // Match the Rank Seperator
            if((*fen == '/')) fen++;
        }   
    }
    // Parsing the side to Move
    
    // To remove the extra whitespace in CLI output
    fen++;
    
    (*fen == 'w') ? (sideToMove = WHITE) : (sideToMove = BLACK) ;

    // Parsing the Castling Rights

    // Moving to the required index w.r.t to side
    fen += 2;
    while(*fen != ' '){
        switch (*fen)
        {
        case 'K':
            castlingRights |= WK;
            break;
        case 'Q':
            castlingRights |= WQ;
            break;
        case 'k':
            castlingRights |= BK;
            break;
        case 'q':
            castlingRights |= BQ;
            break;
        default:
            break;
        }
        // Increment the FEN String
        fen++;
    }

    // Parsing the EnPassant Option
    
    // Increment the FEN String to get to the EnPassant position
    fen++;
    if (*fen != '-'){
        // Parse EnPassant File & Rank
        int file = fen[0] - 'a';
        int rank = 8 - (fen[1] - '0');  

        // Init EnPassant Square
        enPassant = 8 * rank + file;
    }
    else{
        enPassant = NULL_SQUARE;
    }
    // printf("FEN : '%s' \n" , fen);

    // Mapping White Board Ocuupancy
    for(int piece = P; piece <= K; piece++){
        boardOccupancies[WHITE] |= pieceBitBoards[piece];
    }

    // Mapping Black Board Ocuupancy
    for(int piece = p; piece <= k; piece++){
        boardOccupancies[BLACK] |= pieceBitBoards[piece];
    }

    boardOccupancies[BOTH] |= boardOccupancies[WHITE];
    boardOccupancies[BOTH] |= boardOccupancies[BLACK];

    // Hash the position for future use
    hashKey = generateZobristHashKeys();
}

/*
    ATTACKS
*/


// PawnAttacks[side][square]
U64 pawnAttacks[2][64];

// KnightAttacks[square]
U64 knightAttacks[64];

// KingAttacks[square]
U64 kingAttacks[64];

// BishopAttacksMask[square]
U64 bishopAttacksMask[64];

// rookAttacksMask[square]
U64 rookAttacksMask[64];

// bishopAttacks[square][occupancy]
U64 bishopAttacks[64][512];

// rookAttacks[square][occupancy]
U64 rookAttacks[64][4096];
//Generate Pawn Attacks

U64 maskPawnAttacks(int square , int sideToMove){
    // Define Piece BitBoard
    U64 pawnBB = 0ULL;

    // Define the bitBoard for the result of all attacks
    U64 attacksBB = 0ULL;

    // Set Pieces on the board
    setBitOnSquare(pawnBB , square);

    if(!sideToMove){
        // White Pawns
        // Capture forward-Left
        if ((pawnBB >> 7) & maskFileA) attacksBB |= (pawnBB >> 7);
        // Capture forward-Right
        if ((pawnBB >> 9) & maskFileH) attacksBB |= (pawnBB >> 9);
    }
    else{
        // Black pawns
        // Capture Bottom-Right
        if ((pawnBB << 7) & maskFileH) attacksBB |= (pawnBB << 7);
        // Capture Bottom-Left
        if ((pawnBB << 9) & maskFileA) attacksBB |= (pawnBB << 9);
    }
    // Return Attack Map
    return attacksBB;
}

void generatePawnAttacks(){
    for(int square = 0; square < 64; square++){
        pawnAttacks[WHITE][square] = maskPawnAttacks(square , WHITE);
        pawnAttacks[BLACK][square] = maskPawnAttacks(square , BLACK);
    }
}

//Generate Knights Attacks


U64 maskKnightAttacks(int square){
    /*
        Offsets from current square : 
        Top 1 + Left 2 = 10
        Top 1 + Right 2 = 6
        Top 2 + Left 1 = 17
        Top 2 + Right 1 = 15
        Mask accordingly.
    */

    // Define Piece BitBoard
    U64 knightBB = 0ULL;

    // Define the bitBoard for the result of all attacks
    U64 attacksBB = 0ULL;

    // Set Pieces on the board
    setBitOnSquare(knightBB , square);
    if ((knightBB >> 10) & maskFileGH) attacksBB |= (knightBB >> 10);
    if ((knightBB >> 6) & maskFileAB) attacksBB |= (knightBB >> 6);
    if ((knightBB >> 17) & maskFileH) attacksBB |= (knightBB >> 17);
    if ((knightBB >> 15) & maskFileA) attacksBB |= (knightBB >> 15);
    if ((knightBB << 10) & maskFileAB) attacksBB |= (knightBB << 10);
    if ((knightBB << 6) & maskFileGH) attacksBB |= (knightBB << 6);
    if ((knightBB << 17) & maskFileA) attacksBB |= (knightBB << 17);
    if ((knightBB << 15) & maskFileH) attacksBB |= (knightBB << 15);

    // Return Attack Map
    return attacksBB;
}

void generateKnightAttacks(){
    for (int square = 0; square < 64; square++){
        knightAttacks[square] = maskKnightAttacks(square);
    }
}

//Generate King Attacks
U64 maskKingAttacks(int square){
    /*
        Offsets from current square : 
        N = 8
        S = 8
        E = 1
        W = 1
        NE = 9
        SE = 9
        NW = 7
        SW = 7
        Mask accordingly.
    */
    // Define Piece BitBoard
    U64 kingBB = 0ULL;

    // Define the bitBoard for the result of all attacks
    U64 attacksBB = 0ULL;
    // Set Pieces on the board
    setBitOnSquare(kingBB , square);
    if (kingBB >> 8) attacksBB |= (kingBB >> 8); 
    if ((kingBB >> 9) & maskFileH) attacksBB |= (kingBB >> 9); 
    if ((kingBB >> 7) & maskFileA) attacksBB |= (kingBB >> 7); 
    if ((kingBB >> 1) & maskFileH) attacksBB |= (kingBB >> 1); 
    if (kingBB << 8) attacksBB |= (kingBB << 8); 
    if ((kingBB << 9) & maskFileA) attacksBB |= (kingBB << 9); 
    if ((kingBB << 7) & maskFileH) attacksBB |= (kingBB << 7); 
    if ((kingBB << 1) & maskFileA) attacksBB |= (kingBB << 1); 
    // Return Attack Map
    return attacksBB;
}

void generateKingAttacks(){
    for (int square = 0; square < 64; square++){
        kingAttacks[square] = maskKingAttacks(square);
    }
}

//Generate Bishop Occupancy Bits
U64 maskBishopAttacks(int square){
    /*
        Creates a Occupancy BitBoard along the diagonal meaning where all the pieces can occupy to create bounries for this piece other than hard boundries 
        made by rules like a1 (bishop can go beyond that)
    */
    // Define the bitBoard for the result of all attacks
    U64 attacksBB = 0ULL;

    int currRank = square / 8;
    int currFile = square % 8;
    int r , f;
    for(r = currRank + 1 , f = currFile + 1 ; r <= 6 && f <= 6; r++, f++) attacksBB |= (1ULL << (8 * r + f ));
    for(r = currRank - 1 , f = currFile + 1 ; r >= 1 && f <= 6; r--, f++) attacksBB |= (1ULL << (8 * r + f ));
    for(r = currRank + 1 , f = currFile - 1 ; r <= 6 && f >= 1; r++, f--) attacksBB |= (1ULL << (8 * r + f ));
    for(r = currRank - 1 , f = currFile - 1 ; r >= 1 && f >= 1; r--, f--) attacksBB |= (1ULL << (8 * r + f ));
    // Return Attack Map
    return attacksBB;
}

// Generate Bishop Attacks
U64 generateBishopAttacksWithObstacles(int square , U64 boardState){
    // Define the bitBoard for the result of all attacks
    U64 attacksBB = 0ULL;

    int currRank = square / 8;
    int currFile = square % 8;
    int r , f;
    for(r = currRank + 1 , f = currFile + 1 ; r <= 7 && f <= 7; r++, f++){
        attacksBB |= (1ULL << (8 * r + f ));
        if((1ULL << (8 * r + f )) & boardState) break;
    } 
    for(r = currRank - 1 , f = currFile + 1 ; r >= 0 && f <= 7; r--, f++){
        attacksBB |= (1ULL << (8 * r + f ));
        if((1ULL << (8 * r + f )) & boardState) break;
    } 
    for(r = currRank + 1 , f = currFile - 1 ; r <= 7 && f >= 0; r++, f--){
        attacksBB |= (1ULL << (8 * r + f ));
        if((1ULL << (8 * r + f )) & boardState) break;
    } 
    for(r = currRank - 1 , f = currFile - 1 ; r >= 0 && f >= 0; r--, f--){
        attacksBB |= (1ULL << (8 * r + f ));
        if((1ULL << (8 * r + f )) & boardState) break;
    } 
    // Return Attack Map
    return attacksBB;
}

// Generate Rook Attacks
U64 maskRookAttacks(int square){
    // Define the bitBoard for the result of all attacks
    U64 attacksBB = 0ULL;

    int currRank = square / 8;
    int currFile = square % 8;
    int r , f;
    for(r = currRank + 1 ; r <= 6 ; r++) attacksBB |= (1ULL << (8 * r + currFile )); // Bottom Side
    for(r = currRank - 1 ; r >= 1 ; r--) attacksBB |= (1ULL << (8 * r + currFile )); // Top Side
    for(f = currFile + 1 ; f <= 6 ; f++) attacksBB |= (1ULL << (8 * currRank + f )); // Right Side
    for(f = currFile - 1 ; f >= 1 ; f--) attacksBB |= (1ULL << (8 * currRank + f )); // Left Side
    // Return Attack Map
    return attacksBB;
}

U64 generateRookAttacksWithObstacles(int square , U64 boardState){
    /*
        Creates a Occupancy BitBoard along the diagonal meaning where all the pieces can occupy to create bounries for this piece other than hard boundries 
        made by rules like a1 (bishop can go beyond that)
    */
    // Define the bitBoard for the result of all attacks
    U64 attacksBB = 0ULL;

    int currRank = square / 8;
    int currFile = square % 8;
    int r , f;
    for(r = currRank + 1 ; r <= 7 ; r++)
    {
        attacksBB |= (1ULL << (8 * r + currFile )); // Bottom Side
        if((1ULL << (8 * r + currFile)) & boardState) break;
    } 
    for(r = currRank - 1 ; r >= 0 ; r--) 
    {
        attacksBB |= (1ULL << (8 * r + currFile )); // Top Side
        if((1ULL << (8 * r + currFile )) & boardState) break;
    } 
    for(f = currFile + 1 ; f <= 7 ; f++) 
    {
        attacksBB |= (1ULL << (8 * currRank + f )); // Right Side
        if((1ULL << (8 * currRank + f )) & boardState) break;
    } 
    for(f = currFile - 1 ; f >= 0 ; f--) 
    {
        attacksBB |= (1ULL << (8 * currRank + f )); // Left Side
        if((1ULL << (8 * currRank + f )) & boardState) break;
    } 
    // Return Attack Map
    return attacksBB;
}

U64 setOccupancy(int index , int bitCount , U64 attackMask){
    U64 occupancy = 0ULL;
    
    // Loop over the range of bits in the attack mask
    for(int count = 0; count < bitCount; count++){
        
        // Get the index of LS1B
        int square = getLS1BIndex(attackMask);
        
        // Pop LS1B
        popBitOnSquare(attackMask , square);

        // Make sure occupancy is on the board
        if ((index & (1 << count))){
            // Populate Occupancy Map
            occupancy |= (1ULL << square);
        }
    }
    return occupancy;
}

void initBishopAndRookAttacks(int isBishopOrRook){
    for(int square = 0; square < 64; square++){
        // Init Bishop & Rook Masks
        bishopAttacksMask[square] = maskBishopAttacks(square);
        rookAttacksMask[square] = maskRookAttacks(square);

        // Init Current Mask
        U64 attackMask = isBishopOrRook
                            ? bishopAttacksMask[square]
                            : rookAttacksMask[square];

        // Init Relevant Number of bits
        int relevantBits = countBits(attackMask);

        //Init Occupancy Index
        int occupancyIndex = (1 << relevantBits);

        // Loop Over occupancy Indicies
        for(int index = 0; index < occupancyIndex; index++){

            if(isBishopOrRook){
                // Bishop
                
                //Init Current Occupancy variation
                U64 occupancy = setOccupancy(index , relevantBits , attackMask);

                // Init Magic Index
                int magicIndex = (occupancy * bishopMagicNumbers[square]) >> (64 - bishopOccupancyBits[square]);

                // Init Bishop Attacks
                bishopAttacks[square][magicIndex] = generateBishopAttacksWithObstacles(square , occupancy);

            }
            else{
                // Rook

                //Init Current Occupancy variation
                U64 occupancy = setOccupancy(index , relevantBits , attackMask);

                // Init Magic Index
                int magicIndex = (occupancy * rookMagicNumbers[square]) >> (64 - rookOccupancyBits[square]);

                // Init Bishop Attacks
                rookAttacks[square][magicIndex] = generateRookAttacksWithObstacles(square , occupancy);
            }
        }

    }
}

static inline U64 getBishopAttacks(int square , U64 occupancy){
    //Get Bishop Attacks assuming curr board occupancy
    occupancy &= bishopAttacksMask[square];
    occupancy *= bishopMagicNumbers[square];
    occupancy >>= (64 - bishopOccupancyBits[square]);
    
    // Return Bishop Attacks
    return bishopAttacks[square][occupancy];

}

static inline U64 getRookAttacks(int square , U64 occupancy){
    //Get Rook Attacks assuming curr board occupancy
    occupancy &= rookAttacksMask[square];
    occupancy *= rookMagicNumbers[square];
    occupancy >>= (64 - rookOccupancyBits[square]);

    // Return Rook Attacks
    return rookAttacks[square][occupancy];
}

static inline U64 getQueenAttacks(int square , U64 occupancy){

    // Return Rook Attacks
    return (getBishopAttacks(square, occupancy) | getRookAttacks(square, occupancy));;
}

U64 generateMagicNumberCandidates(){
    return getRandomU64Number() & getRandomU64Number() & getRandomU64Number();
}

/*
    MAGIC NUMBERS
*/

U64 getMagicNumber(int square , int relevantBits , int isBishopOrRook){
    /*
        Max in Bishop Occupancy is 9 , so 2^9 = 512 occupancy possibilities
        Max in Rook Occupancy is 12 , so 2^12 = 4096 occupancy possibilities
    */

   // Init Occupancy map
    U64 occupancyMap[4096];

    // Init Attack tables
    U64 attacks[4096];

    // Used attacks table
    U64 usedAttacks[4096];

    // Init Attack Mask for current piece
    U64 attackMask = isBishopOrRook 
                    ? maskBishopAttacks(square) 
                    : maskRookAttacks(square);
    // Init Occupancy indices
    int occupancyIndex = 1 << relevantBits;

    // Loop over occupancy index
    for (int index = 0; index < occupancyIndex; index++){
        // Init Occupancy Map
        occupancyMap[index] = setOccupancy(index , relevantBits , attackMask);

        // Set Attacks
        attacks[index] = isBishopOrRook 
                            ? generateBishopAttacksWithObstacles(square , occupancyMap[index])
                            : generateRookAttacksWithObstacles(square , occupancyMap[index]); 
    }

    // Test Magic Number Candidates
    for (int randomCount = 0; randomCount < 100000000; randomCount++){
        // Generate Magic Number Candidate
        U64 magicNumberCandidate = generateMagicNumberCandidates();

        // Skip Inappropriate Magic Numbers
        if (countBits((attackMask * magicNumberCandidate) & 0xFF00000000000000ULL) < 6) continue;

        // Init Used Attack 
        memset(usedAttacks , 0ULL , sizeof(usedAttacks)); // Set N bytes of S to C in memset(s , c , n)

        // Init Index & Fail Flag
        int index, failFlag;
        
        // Test Magic Index
        for(index = 0 , failFlag = 0; !failFlag && index < occupancyIndex; index++){
            // Init Magic Index
            int magicIndex = (int)((occupancyMap[index] * magicNumberCandidate) >> (64 - relevantBits));

            // If Magic Index Works
            if (usedAttacks[magicIndex] == 0ULL){
                // init Used Attacks
                usedAttacks[magicIndex] = attacks[index];
            }
            else if (usedAttacks[magicIndex] != attacks[index]){
                // Magic Index doesn't work
                failFlag = 1;
            }
        }
        if (!failFlag){
            return magicNumberCandidate;
        }
    }

    // If Not Fount
    printf(" Magic Number Not Found !!! \n");
    return 0ULL;
    
}

void initMagicNumbers(){
    // Loop over all squares
    for(int square = 0; square < 64; square++){
        // Init Rook Magic Numbers  
        rookMagicNumbers[square] = getMagicNumber(square , rookOccupancyBits[square] , rook);
        printf(" 0x%llxULL ,\n" , rookMagicNumbers[square]);
    }
    for(int square = 0; square < 64; square++){
        // Init Bishop Magic Numbers  
        bishopMagicNumbers[square] = getMagicNumber(square , bishopOccupancyBits[square] , bishop);
        // printf(" 0x%llxULL ,\n" , bishopMagicNumbers[square]);
    }
}


/*
    MOVE FORMATTING -> (6 * 2 (for src and dest squares) + 4 (12 pieces) + 4 (1 bit for each promoting option) + 4(1 for each flag) = 21)
           Binary(Max Values)               Mapping           Hexadecimal Values
    0000 0000 0000 0000 0011 1111         Source Square             0x3F
    0000 0000 0000 1111 1100 0000         Target Square             0xFC0
    0000 0000 1111 0000 0000 0000         Piece                     0xF000
    0000 1111 0000 0000 0000 0000         Promoted Piece            0xF0000
    0001 0000 0000 0000 0000 0000         Capture Flag              0x100000
    0010 0000 0000 0000 0000 0000         Double Push Flag          0x200000
    0100 0000 0000 0000 0000 0000         EnPassant Flag            0x400000
    1000 0000 0000 0000 0000 0000         Castling Flag             0x800000
    
*/
#define encodeMove(source, target, piece, promoted, capture, double, enpassant, castling) \
            (source) |      \
            (target << 6) |  \
            (piece << 12) |   \
            (promoted << 16) | \
            (capture << 20) |   \
            (double << 21) |     \
            (enpassant << 22) |   \
            (castling << 23)       \
             
#define extractSource(move)         (move & 0x3F)
#define extractTarget(move)         ((move & 0xFC0) >> 6)
#define extractMovedPiece(move)     ((move & 0xF000) >> 12)
#define extractPromotedPiece(move)  ((move & 0xF0000) >> 16)
#define extractCaptureFlag(move)    ((move & 0x100000) >> 20)
#define extractDoublePushFlag(move) ((move & 0x200000) >> 21)
#define extractEnPassantFlag(move)  ((move & 0x400000) >> 22)
#define extractCastlingFlag(move)   ((move & 0x800000) >> 23)

typedef struct {
    // Moves
    int moves[256]; // Keeping Track of 256 past moves at any time
    int moveCount; // Treat it like the index for the above array
}Move;

// Print Move (For UCI Purposes)
void printMove(int move){
    // Allignment according to the requirements of UCI protocol
    if(extractPromotedPiece(move)){
        printf("%s%s%c", squareToCoordinates[extractSource(move)],
                        squareToCoordinates[extractTarget(move)],
                        promotedPieces[extractPromotedPiece(move)]);
    }
    else{
        printf("%s%s", squareToCoordinates[extractSource(move)],
                        squareToCoordinates[extractTarget(move)]);
    }
}

static inline void addMove(Move * moveList , int move){
    // Store Move
    moveList->moves[moveList->moveCount] = move; 
    // Increment Move Count
    moveList->moveCount++;
}

void printMoveList(Move * moveList){

    if (!moveList->moveCount){
        printf(" No Move in Move List \n");
        return;
    }

    printf("\n\t Move  Piece , Capture , Double , Enpassant , Castling\n");
    for(int count = 0; count < moveList->moveCount; count++){
        int move = moveList->moves[count];  
        #ifdef WIN64
        printf("\t%s-%s%c   %s        %d        %d          %d          %d\n", squareToCoordinates[extractSource(move)],
                        squareToCoordinates[extractTarget(move)],
                        extractPromotedPiece(move) ? promotedPieces[extractPromotedPiece(move)] : ' ',
                        asciiPieces[extractMovedPiece(move)],
                        extractCaptureFlag(move) ? 1 : 0,
                        extractDoublePushFlag(move) ? 1 : 0,
                        extractEnPassantFlag(move) ? 1 : 0,
                        extractCastlingFlag(move) ? 1 : 0);
        #else 
        printf("\t%s-%s%c   %s        %d        %d          %d          %d\n", squareToCoordinates[extractSource(move)],
                        squareToCoordinates[extractTarget(move)],
                        extractPromotedPiece(move) ? promotedPieces[extractPromotedPiece(move)] : ' ',
                        unicodePieces[extractMovedPiece(move)],
                        extractCaptureFlag(move) ? 1 : 0,
                        extractDoublePushFlag(move) ? 1 : 0,
                        extractEnPassantFlag(move) ? 1 : 0,
                        extractCastlingFlag(move) ? 1 : 0);
        #endif

    }
    printf("\n Total Number Of Moves : %d\n" , moveList->moveCount);
}

/*
    COPY/MAKE APPROACH
    // memcpy(Dest , src , sizeof_bits_to_copy) // sizeof(pieceBitBoards) = 96 
    // sizeof(boardOccupancy) = 24 
*/

#define copyBoardState()                                                                            \
    U64 pieceBitBoardsCopy[12] , boardOccupanciesCopy[3];                                           \
    int sideToMoveCopy , enPassantCopy , castlingRightsCopy;                                        \
    memcpy(pieceBitBoardsCopy , pieceBitBoards , 96);                                               \
    memcpy(boardOccupanciesCopy , boardOccupancies , 24);                                           \
    sideToMoveCopy = sideToMove , enPassantCopy = enPassant , castlingRightsCopy = castlingRights;  \
    U64 hashKeyCopy = hashKey;                                                                      \

#define restoreBoardState()                                                                         \
    memcpy(pieceBitBoards , pieceBitBoardsCopy , 96);                                               \
    memcpy(boardOccupancies , boardOccupanciesCopy , 24);                                           \
    sideToMove = sideToMoveCopy , enPassant = enPassantCopy , castlingRights = castlingRightsCopy;  \
    hashKey = hashKeyCopy;                                                                          \

/*
    MOVE GENERATOR
*/

static inline int isSquareAttacked(int square , int side){

    // Attacked by White Pawns
    if((side == WHITE) && (pawnAttacks[BLACK][square] & pieceBitBoards[P])) return 1;

    // Attacked by BLACK Pawns
    if((side == BLACK) && (pawnAttacks[WHITE][square] & pieceBitBoards[p])) return 1;

    // Attacked by Knight
    if (knightAttacks[square] & ((side == WHITE) ? pieceBitBoards[N] : pieceBitBoards[n])) return 1;

    // Attacked by King
    if (kingAttacks[square] & ((side == WHITE) ? pieceBitBoards[K] : pieceBitBoards[k])) return 1;
    
    // Attacked by Bishop
    if (getBishopAttacks(square , boardOccupancies[BOTH]) & ((side == WHITE) ? pieceBitBoards[B] : pieceBitBoards[b])) return 1;
    
    // Attacked by Rooks
    if (getRookAttacks(square , boardOccupancies[BOTH]) & ((side == WHITE) ? pieceBitBoards[R] : pieceBitBoards[r])) return 1;
    
    // Attacked by Rooks
    if (getQueenAttacks(square , boardOccupancies[BOTH]) & ((side == WHITE) ? pieceBitBoards[Q] : pieceBitBoards[q])) return 1;

    // Return False By default
    return 0;
}

void printAllAttackedSquares(int side){
    printf("\n");
    for(int rank = 0; rank < 8; rank++){
        for(int file = 0; file < 8; file++){
            int square = 8 * rank + file;

            if(!file) printf(" %d " , (8-rank));
            // check If Current Square is Attacked
            printf(" %d" , isSquareAttacked(square , side) ? 1 : 0);
        }
        printf("\n");
    }
    printf("\n    a b c d e f g h \n");
}

static inline void generateMoves(Move * moveList){
    /*
        This generates Pseudo Legal Moves
    */
    
    // Init Move List
    moveList->moveCount = 0;

    // Define Src and Dest/Target Squares Index
    int srcSquare , targetSquare;

    // Define Current piece's BitBoard copy & its attacks -> Used to generate the target square.
    U64 bitBoardCopyOfPiece , attacks;

    // Loop over all the bitBoards 
    for(int piece = P; piece <= k; piece++){
        // Init Piece BitBoard Copy to that of piece's
        bitBoardCopyOfPiece = pieceBitBoards[piece];

        if (sideToMove == WHITE){
            // Generate White Pawns & White King Castling Moves

            // Identify White pawn index from its bitBoard
            if (piece == P){
                // Loop over White Pawns BitBoard
                while(bitBoardCopyOfPiece){
                    // While this is not a empty bitboard == 0ULL
                    srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

                    // Init Target Square
                    targetSquare = srcSquare - 8; // due to board definition in enum

                    /*
                        Pawn moves are of 3 types : 
                            1) Pawn Promotion 
                            2) 1 square forward ( possible anytime ) 
                            3) 2 sqaures forward ( only for 1st move of the pawn )
                    */

                    // generate Quite Pawn Moves
                    if(!(targetSquare < a8) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare))){
                        // This condition is true only if the target square is on board and it is not occupied by any other piece.

                        // handle pawn promotion
                        if((srcSquare >= a7) && (srcSquare <= h7)){

                            // Debug Purposes                            
                            // printf("Pawn Promotion: %s to %s , promote to Q\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion: %s to %s , promote to R\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion: %s to %s , promote to B\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion: %s to %s , promote to N\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                            // Add Move to a Move list
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , Q , 0 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , R , 0 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , B , 0 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , N , 0 , 0 , 0 , 0));
                        }
                        else{
                            // handle 1 square forward pawn move

                            // Debug Purposes
                            // printf("Pawn Push by 1: %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            
                            // Add Move to a Move list
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0 , 0));

                            // handle 2 squares forward pawn move
                            if(((srcSquare >= a2) && (srcSquare <= h2)) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare - 8))){

                                // Debug Purposes
                                // printf("Pawn Push by 2: %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare - 8]);

                                // Add Move to a Move list
                                addMove(moveList , encodeMove(srcSquare , targetSquare - 8 , piece , 0 , 0 , 1 , 0 , 0));
                            }   
                        }

                    }

                    // Init Pawn Attacks BitBoard
                    attacks = pawnAttacks[sideToMove][srcSquare] & boardOccupancies[BLACK];

                    // Generate pawn Captures
                    while (attacks){
                        // init target square
                        targetSquare = getLS1BIndex(attacks);
                        // handle pawn promotion with Capture
                        if((srcSquare >= a7) && (srcSquare <= h7)){
                            
                            // Debug Purposes
                            // printf("Pawn Promotion with Capture: %s to %s , promote to Q\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion with Capture: %s to %s , promote to R\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion with Capture: %s to %s , promote to B\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion with Capture: %s to %s , promote to N\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                            // Add Move to a Move list
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , Q , 1 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , R , 1 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , B , 1 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , N , 1 , 0 , 0 , 0));
                        }
                        else{
                            // handle 1 square forward pawn move

                            // Debug Purposes
                            // printf("Pawn Push by 1 with Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                            // Add Move to a Move list
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0 , 0));

                        }

                        // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                        popBitOnSquare(attacks , targetSquare);
                    }

                    // generate EnPassant Capture
                    if (enPassant != NULL_SQUARE){
                        //
                        U64 enpassantAttacks = pawnAttacks[sideToMove][srcSquare] & (1ULL << enPassant);
                        if (enpassantAttacks){
                            // Condition true if only a enPassant is available

                            // init EnPassant target Square
                            int enPassantTargetSquare = getLS1BIndex(enpassantAttacks);
                            
                            // Debug Purposes
                            // printf("EnPassant Pawn Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[enPassantTargetSquare]);

                            // Add Move to a Move list
                            addMove(moveList , encodeMove(srcSquare , enPassantTargetSquare , piece , 0 , 1 , 0 , 1 , 0));
                        }
                    }
                    // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                    popBitOnSquare(bitBoardCopyOfPiece , srcSquare);

                } 
            }

            if(piece == K){
                // Kingside Castling
                if(castlingRights & WK){
                    // Make sure there is no piece btw king and rook on the kingside
                    if(!(getBitOnSquare(boardOccupancies[BOTH] , f1)) && !(getBitOnSquare(boardOccupancies[BOTH] , g1))){
                        // Make sure the king and f1 are not under attack
                        if(!isSquareAttacked(e1 , BLACK) & !isSquareAttacked(f1 , BLACK)){

                            // Debug Purposes
                            // printf("Castling Kingside Move : e1 to g1 \n");
                            // Add Move to a Move list
                            addMove(moveList , encodeMove(e1 , g1 , piece , 0 , 0 , 0 , 0, 1));
                        }
                    }
                }

                // Queenside Castling
                if(castlingRights & WQ){
                    // Make sure there is no piece btw king and rook on the queenside
                    if(!(getBitOnSquare(boardOccupancies[BOTH] , b1)) && !(getBitOnSquare(boardOccupancies[BOTH] , c1)) && !(getBitOnSquare(boardOccupancies[BOTH] , d1))){
                        // Make sure the king and d1 are not under attack
                        if(!isSquareAttacked(e1 , BLACK) & !isSquareAttacked(d1 , BLACK)){

                            // Debug Purposes
                            // printf("Castling Queenside Move : e1 to c1 \n");

                            // Add Move to a Move list
                            addMove(moveList , encodeMove(e1 , c1 , piece , 0 , 0 , 0 , 0, 1));
                        }
                    }
                }
            }
                
        }
        else{
            // Generate Black Pawns & Black King Castling Moves
            if (piece == p){
                // Loop over Black Pawns BitBoard
                while(bitBoardCopyOfPiece){
                    // While this is not a empty bitboard == 0ULL
                    srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

                    // Init Target Square
                    targetSquare = srcSquare + 8; // due to board definition in enum

                    /*
                        Quite Pawn (No Capure Moves) moves are of 3 types : 
                            1) Pawn Promotion 
                            2) 1 square forward ( possible anytime ) 
                            3) 2 sqaures forward ( only for 1st move of the pawn )
                    */

                    // generate Quite Pawn Moves
                    if(!(targetSquare > h1) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare))){
                        // This condition is true only if the target square is on board and it is not occupied by any other piece.

                        // handle pawn promotion
                        if((srcSquare >= a2) && (srcSquare <= h2)){
                            
                            // Debug Purposes
                            // printf("Pawn Promotion: %s to %s , promote to Q\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion: %s to %s , promote to R\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion: %s to %s , promote to B\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion: %s to %s , promote to N\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                            // Add Move to a Move list
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , q , 0 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , r , 0 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , b , 0 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , n , 0 , 0 , 0 , 0));
                        }
                        else{
                            // handle 1 square forward pawn move
                            // printf("Pawn Push by 1: %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0 , 0));

                            // handle 2 squares forward pawn move
                            if(((srcSquare >= a7) && (srcSquare <= h7)) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare + 8))){
                                // printf("Pawn Push by 2: %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare + 8]);

                                addMove(moveList , encodeMove(srcSquare , targetSquare + 8 , piece , 0 , 0 , 1 , 0 , 0));
                            }   
                        }

                    }

                    // Init Pawn Attacks BitBoard
                    attacks = pawnAttacks[sideToMove][srcSquare] & boardOccupancies[WHITE];

                    // Generate pawn Captures
                    while (attacks){
                        // init target square
                        targetSquare = getLS1BIndex(attacks);
                        // handle pawn promotion with Capture
                        if((srcSquare >= a2) && (srcSquare <= h2)){
                            
                            // printf("Pawn Promotion with Capture: %s to %s , promote to Q\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion with Capture: %s to %s , promote to R\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion with Capture: %s to %s , promote to B\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                            // printf("Pawn Promotion with Capture: %s to %s , promote to N\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                            // Add Move to a Move list
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , q , 1 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , r , 1 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , b , 1 , 0 , 0 , 0));
                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , n , 1 , 0 , 0 , 0));
                        }
                        else{
                            // handle 1 square forward pawn move
                            // printf("Pawn Push by 1 with Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                            addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0 , 0));

                        }

                        // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                        popBitOnSquare(attacks , targetSquare);
                    }

                    // generate EnPassant Capture
                    if (enPassant != NULL_SQUARE){
                        //
                        U64 enpassantAttacks = pawnAttacks[sideToMove][srcSquare] & (1ULL << enPassant);
                        if (enpassantAttacks){
                            // Condition true if only a enPassant is available

                            // init EnPassant target Square
                            int enPassantTargetSquare = getLS1BIndex(enpassantAttacks);
                            // printf("EnPassant Pawn Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[enPassantTargetSquare]);

                            addMove(moveList , encodeMove(srcSquare , enPassantTargetSquare , piece , 0 , 1 , 0 , 1 , 0));
                        }
                    }

                    // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                    popBitOnSquare(bitBoardCopyOfPiece , srcSquare);

                } 
            }
            if(piece == k){
                // Kingside Castling
                if(castlingRights & BK){
                    // Make sure there is no piece btw king and rook on the kingside
                    if(!(getBitOnSquare(boardOccupancies[BOTH] , f8)) && !(getBitOnSquare(boardOccupancies[BOTH] , g8))){
                        // Make sure the king and f8 are not under attack
                        if(!isSquareAttacked(e8 , WHITE) & !isSquareAttacked(f8 , WHITE)){
                            // printf("Castling Kingside Move : e8 to g8 \n");

                            addMove(moveList , encodeMove(e8 , g8 , piece , 0 , 0 , 0 , 0, 1));
                        }
                    }
                }

                // Queenside Castling
                if(castlingRights & BQ){
                    // Make sure there is no piece btw king and rook on the queenside
                    if(!(getBitOnSquare(boardOccupancies[BOTH] , b8)) && !(getBitOnSquare(boardOccupancies[BOTH] , c8)) && !(getBitOnSquare(boardOccupancies[BOTH] , d8))){
                        // Make sure the king and d8 are not under attack
                        if(!isSquareAttacked(e8 , WHITE) & !isSquareAttacked(d8 , WHITE)){
                            // printf("Castling Queenside Move : e8 to c8 \n");

                            addMove(moveList , encodeMove(e8 , c8 , piece , 0 , 0 , 0 , 0, 1));
                        }
                    }
                }
            }
            
        }

        // Generate Knight Moves
        if((sideToMove == WHITE) ? (piece == N) : (piece == n)){
            // Loop over src Square of piece's BitBoard copy
            while (bitBoardCopyOfPiece)
            {
                // Init Src Square 
                srcSquare = getLS1BIndex(bitBoardCopyOfPiece);
                
                // Init Piece Attacks 
                
                // ~boardOccupancies[WHITE] => 1 if white pieces are not present else 0 , ~ = Bitwise NOT 
                attacks = knightAttacks[srcSquare] & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

                // Loop over the target Squares available from the generated attacks 
                while (attacks){
                    // init target square
                    targetSquare = getLS1BIndex(attacks);
                    
                    if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
                        // Quite Moves
                        // printf("Knight Move : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
                    }
                    else{
                        // Capture Moves
                        // printf("Knight Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
                    }

                    // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                    popBitOnSquare(attacks , targetSquare);
                }
                // Pop LS1B from the copy
                popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
            }
            
        }

        // Generate Bishop Moves
        if((sideToMove == WHITE) ? (piece == B) : (piece == b)){
            // Loop over src Square of piece's BitBoard copy
            while (bitBoardCopyOfPiece)
            {
                // Init Src Square 
                srcSquare = getLS1BIndex(bitBoardCopyOfPiece);
                
                // Init Piece Attacks 
                // ~boardOccupancies[WHITE] => 1 if white pieces are not present else 0 , ~ = Bitwise NOT 
                attacks = getBishopAttacks(srcSquare , boardOccupancies[BOTH]) & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

                // Loop over the target Squares available from the generated attacks 
                while (attacks){
                    // init target square
                    targetSquare = getLS1BIndex(attacks);
                    
                    if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
                        // Quite Moves
                        // printf("Bishop Move : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
                    }
                    else{
                        // Capture Moves
                        // printf("Bishop Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                        
                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
                    }

                    // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                    popBitOnSquare(attacks , targetSquare);
                }
                // Pop LS1B from the copy
                popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
            }
            
        }

        // Generate Rook Moves
        if((sideToMove == WHITE) ? (piece == R) : (piece == r)){
            // Loop over src Square of piece's BitBoard copy
            while (bitBoardCopyOfPiece)
            {
                // Init Src Square 
                srcSquare = getLS1BIndex(bitBoardCopyOfPiece);
                
                // Init Piece Attacks 
                
                // ~boardOccupancies[WHITE] => 1 if white pieces are not present else 0 , ~ = Bitwise NOT 
                attacks = getRookAttacks(srcSquare , boardOccupancies[BOTH]) & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

                // Loop over the target Squares available from the generated attacks 
                while (attacks){
                    // init target square
                    targetSquare = getLS1BIndex(attacks);
                    
                    if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
                        // Quite Move 
                        // printf("Rook Move : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
                    }
                    else{
                        // Quite Moves
                        // printf("Rook Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                        
                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
                    }

                    // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                    popBitOnSquare(attacks , targetSquare);
                }
                // Pop LS1B from the copy
                popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
            }
            
        }

        // Generate Queen Moves
        if((sideToMove == WHITE) ? (piece == Q) : (piece == q)){
            // Loop over src Square of piece's BitBoard copy
            while (bitBoardCopyOfPiece)
            {
                // Init Src Square 
                srcSquare = getLS1BIndex(bitBoardCopyOfPiece);
                
                // Init Piece Attacks 
                
                // ~boardOccupancies[WHITE] => 1 if white pieces are not present else 0 , ~ = Bitwise NOT 
                attacks = getQueenAttacks(srcSquare , boardOccupancies[BOTH]) & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

                // Loop over the target Squares available from the generated attacks 
                while (attacks){
                    // init target square
                    targetSquare = getLS1BIndex(attacks);
                    
                    if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
                        // Quite Moves
                        // printf("Queen Move : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
                    }
                    else{
                        // Capture Moves
                        // printf("Queen Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                        
                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
                    }

                    // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                    popBitOnSquare(attacks , targetSquare);
                }
                // Pop LS1B from the copy
                popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
            }
            
        }

        // Generate King Moves
        if((sideToMove == WHITE) ? (piece == K) : (piece == k)){
            // Loop over src Square of piece's BitBoard copy
            while (bitBoardCopyOfPiece)
            {
                // Init Src Square 
                srcSquare = getLS1BIndex(bitBoardCopyOfPiece);
                
                // Init Piece Attacks 
                
                // ~boardOccupancies[WHITE] => 1 if white pieces are not present else 0 , ~ = Bitwise NOT 
                attacks = kingAttacks[srcSquare] & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

                // Loop over the target Squares available from the generated attacks 
                while (attacks){
                    // init target square
                    targetSquare = getLS1BIndex(attacks);
                    
                    if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
                        // Quite Moves
                        // printf("King Move : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);

                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
                    }
                    else{
                        // Capture Moves
                        // printf("King Capture : %s to %s\n" , squareToCoordinates[srcSquare] , squareToCoordinates[targetSquare]);
                        
                        addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
                    }

                    // pop LS1B from bitBoard Copy -> To Simulate Movement to that square
                    popBitOnSquare(attacks , targetSquare);
                }
                // Pop LS1B from the copy
                popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
            }
            
        }
    }
}

/*
    CASTLING PRECOMPUTED MASK

    UPDATING CASTLING RIGHTS
            MoveType               Castling Right(Before)          Castling Rights(After)      Hex Constants to get from Before to After
    1) K/k & R/r didn't move            1111                            1111                                0xF(=15)
    2) K Moved                          1111                            1100                                0xC(=12)
    3) Kingside R Moved                 1111                            1110                                0xE(=14)
    4) Queenside R Moved                1111                            1101                                0xD(=13)
    5) k Moved                          1111                            0011                                0x3(=3)
    6) Kingside r Moved                 1111                            1011                                0xB(=11)
    7) Queenside r Moved                1111                            0111                                0x7(=7)
*/

const int castlingRightsMask[64] = {
     7 , 15 , 15 , 15 ,  3 , 15 , 15 , 11 ,
    15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
    15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
    15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
    15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
    15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
    15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
    13 , 15 , 15 , 15 , 12 , 15 , 15 , 14 
};

// Make Move on a Board(can be a copy or the original Board)
static inline int makeMove(int move , int moveFlag){
    if (moveFlag == allMoves){
        // Quite Moves  

        // Preserve Board State
        copyBoardState();

        // parse the move
        int srcSquare = extractSource(move);
        int targetSquare = extractTarget(move);
        int pieceMoved = extractMovedPiece(move);
        int promotedPiece = extractPromotedPiece(move);
        int captureFlag = extractCaptureFlag(move);
        int doublePushFlag = extractDoublePushFlag(move);
        int enPassantFlag = extractEnPassantFlag(move);
        int castlingFlag = extractCastlingFlag(move);

        // Move Piece
        popBitOnSquare(pieceBitBoards[pieceMoved] , srcSquare);
        setBitOnSquare(pieceBitBoards[pieceMoved] , targetSquare);

        // Hash Piece -> remove piece from sc square and put on target square
        hashKey ^= pieceKeys[pieceMoved][srcSquare]; // Remove the piece from the hash
        hashKey ^= pieceKeys[pieceMoved][targetSquare]; // Put the piece on required square and put it in hash

        // Handle Capture Moves
        if(captureFlag){
            // The move actually captures something

            // Pick Up BitBoard piece index ranges depending on side

            // Defines what ranges to iterate over , start to end
            int startPiece , endPiece; 

            if(sideToMove == WHITE){
                startPiece = p;
                endPiece = k;
            }
            else{
                startPiece = P;
                endPiece = K;
            }
            
            // Loop Over the opponents(other side to move)'s bitboard 
            for(int consideredPiece = startPiece ; consideredPiece <= endPiece ; consideredPiece++){
                // If the current piece type is present on the target square
                if(getBitOnSquare(pieceBitBoards[consideredPiece] , targetSquare)){
                    // Remove the corresponding piece from the target square 
                    popBitOnSquare(pieceBitBoards[consideredPiece] , targetSquare);

                    // Remove the piece from the hash key
                    hashKey ^= pieceKeys[consideredPiece][targetSquare];
                    break; // only 1 piece possible for a square (acc to rules) => end the search
                }                
            }
        }

        // Handling Promotion cases
        if(promotedPiece){
            // Erase the pawn from the target square
            // popBitOnSquare(pieceBitBoards[(sideToMove == WHITE) ? P : p] , targetSquare);

            if(sideToMove == WHITE){
                popBitOnSquare(pieceBitBoards[P] , targetSquare);

                // Remove the Pawn from the Hash
                hashKey ^= pieceKeys[P][targetSquare];
            }
            else{
                popBitOnSquare(pieceBitBoards[p] , targetSquare);
                
                // Remove the Pawn from the Hash
                hashKey ^= pieceKeys[p][targetSquare];
            }

            // Place the promoted piece on the target square
            setBitOnSquare(pieceBitBoards[promotedPiece] , targetSquare);
            
            // Put the promoted piece in the Hash
            hashKey ^= pieceKeys[promotedPiece][targetSquare];
        }

        // Handling EnPassant Capture Cases
        if(enPassantFlag){
            // Depending on the side erase the correct pawn
            // (sideToMove == WHITE) 
            //                     ? popBitOnSquare(pieceBitBoards[p] , targetSquare + 8)
            //                     : popBitOnSquare(pieceBitBoards[P] , targetSquare - 8);

            if (sideToMove == WHITE){
                popBitOnSquare(pieceBitBoards[p] , targetSquare + 8);

                // Remove the Pawn Captured from the hash
                hashKey ^= pieceKeys[p][targetSquare + 8];
            }
            else{
                popBitOnSquare(pieceBitBoards[P] , targetSquare - 8);
                
                // Remove the Pawn Captured from the hash
                hashKey ^= pieceKeys[P][targetSquare - 8];
            }
        }

        // Hash Enpassant if available 
        if (enPassant != NULL_SQUARE) hashKey ^= enPassantKeys[enPassant];

        enPassant = NULL_SQUARE;

        // Handling Double Push created EnPassant
        if(doublePushFlag){
            // Set enPassant Square depending on the side
            // (sideToMove == WHITE) 
            //                     ? (enPassant = targetSquare + 8)
            //                     : (enPassant = targetSquare - 8);

            if (sideToMove == WHITE){
                // Set enpassant sqaure
                enPassant = targetSquare + 8;

                // Hash the enpassant into the new position hash
                hashKey ^= enPassantKeys[targetSquare + 8];
            }
            else{
                enPassant = targetSquare - 8;
                hashKey ^= enPassantKeys[targetSquare - 8];
            }
        }

        // Handle Castling Moves
        // For hashing , castling is king move , so remove rook position from hash and update it by adding the new rook position 

        if(castlingFlag){
            switch (targetSquare)
            {   
                // White Castles Kingside
                case (g1):
                    // Move H Rook to f1
                    popBitOnSquare(pieceBitBoards[R] , h1);
                    setBitOnSquare(pieceBitBoards[R] , f1);
                    hashKey ^= pieceKeys[R][h1]; // Remove Rook from h1
                    hashKey ^= pieceKeys[R][f1]; // Put Rook from f1
                    break;
                // White Castles Queenside
                case (c1):
                    // Move H Rook to f1
                    popBitOnSquare(pieceBitBoards[R] , a1);
                    setBitOnSquare(pieceBitBoards[R] , d1);
                    hashKey ^= pieceKeys[R][a1]; // Remove Rook from a1
                    hashKey ^= pieceKeys[R][d1]; // Put Rook from d1
                    break;
                // Black Castles Kingside
                case (g8):
                    // Move H Rook to f1
                    popBitOnSquare(pieceBitBoards[r] , h8);
                    setBitOnSquare(pieceBitBoards[r] , f8);
                    hashKey ^= pieceKeys[r][h8]; // Remove Rook from h8
                    hashKey ^= pieceKeys[r][f8]; // Put Rook from f8
                    break;
                // Black Castles Queenside
                case (c8):
                    // Move H Rook to f1
                    popBitOnSquare(pieceBitBoards[r] , a8);
                    setBitOnSquare(pieceBitBoards[r] , d8);
                    hashKey ^= pieceKeys[r][a8]; // Remove Rook from a8
                    hashKey ^= pieceKeys[r][d8]; // Put Rook from d8
                    break; 
                default:
                    break;
            }
        }

        // Hash the castling rights into the hash , firts remove all rights and then add the current rights
        hashKey ^= castlingKeys[castlingRights];

        // Update Castling Rights
        castlingRights &= castlingRightsMask[srcSquare];
        castlingRights &= castlingRightsMask[targetSquare];
        hashKey ^= castlingKeys[castlingRights]; // Add the new Rights

        // Update board Occupancies
        memset(boardOccupancies , 0ULL , 24);

        // Update White Side
        for(int piece = P; piece <= K; piece++){
            boardOccupancies[WHITE] |= pieceBitBoards[piece];
        }
        
        // Update Black Side
        for(int piece = p; piece <= k; piece++){
            boardOccupancies[BLACK] |= pieceBitBoards[piece];
        }

        // Update Full Boar Occupancy
        boardOccupancies[BOTH] |= boardOccupancies[WHITE];
        boardOccupancies[BOTH] |= boardOccupancies[BLACK];

        // Change the side after the move
        sideToMove ^= 1; // Same as NOT

        // Include the change of sides to the hash key
        hashKey ^= sideToMoveKey;

        //
        // =========================== DEBUG HASH KEY INCREMENTAL UPDATE ===========================
        //

        // Build the hash for the new position reached after the move is made
        // U64 newPostionHashKey = generateZobristHashKeys();

        // // In Case the built hash key is not the same as the hash key generated by incremental updates , we interrupt execution
        // if (hashKey != newPostionHashKey){
        //     printf("Make Move : \n");
        //     printf("Move : ");
        //     printMove(move);
        //     printBoard();
        //     printf("Hash key should be : %llx\n" , newPostionHashKey);
        //     getchar();
        // }

        // Make sure the king is not in check ,i.e, not attacked
        if(isSquareAttacked((sideToMove == WHITE) ? getLS1BIndex(pieceBitBoards[k]) : getLS1BIndex(pieceBitBoards[K]), sideToMove)){
            
            // If condition True => King is in check => Move is illegal
            restoreBoardState();

            // Illegal Move
            return 0;
        }
        else{
            // Legal Move
            return 1;
        }

    }
    else{
        // Capture Moves
        if (extractCaptureFlag(move)){
            // Make sure the move is a capture
            makeMove(move , allMoves);
        }
        else{
            // Move is not a capture , do not make the move => Illegal Move
            return 0;
        }
    }
}

/*
    EVALUATION
*/

/*
    ♙ =   100   = ♙
    ♘ =   300   = ♙ * 3
    ♗ =   350   = ♙ * 3 + ♙ * 0.5
    ♖ =   500   = ♙ * 5
    ♕ =   1000  = ♙ * 10
    ♔ =   10000 = ♙ * 100
    
*/

// Define bounds to present mating scores in UCI
#define INFINITY 50000
#define MATE_VALUE 49000
#define MATE_SCORE 48000

// int materialScore[12] = {
//     100, // Value of W pawn
//     300, // Value of W Knight
//     350, // Value of W Bishop
//     500, // Value of W Rook
//    1000, // Value of W Queen
//   10000, // Value of W King
//    -100, // Value of B pawn
//    -300, // Value of B Knight
//    -350, // Value of B Bishop
//    -500, // Value of B Rook
//   -1000, // Value of B Queen
//  -10000, // Value of B King
// };

// pawn positional score
const int positionalPawnScore[64] = 
{
    90,  90,  90,  90,  90,  90,  90,  90,
    30,  30,  30,  40,  40,  30,  30,  30,
    20,  20,  20,  30,  30,  30,  20,  20,
    10,  10,  10,  20,  20,  10,  10,  10,
     5,   5,  10,  20,  20,   5,   5,   5,
     0,   0,   0,   5,   5,   0,   0,   0,
     0,   0,   0, -10, -10,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0
};

// knight positional score
const int positionalKnightScore[64] = 
{
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,  10,  10,   0,   0,  -5,
    -5,   5,  20,  20,  20,  20,   5,  -5,
    -5,  10,  20,  30,  30,  20,  10,  -5,
    -5,  10,  20,  30,  30,  20,  10,  -5,
    -5,   5,  20,  10,  10,  20,   5,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5, -10,   0,   0,   0,   0, -10,  -5
};

// bishop positional score
const int positionalBishopScore[64] = 
{
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,  10,  10,   0,   0,   0,
     0,   0,  10,  20,  20,  10,   0,   0,
     0,   0,  10,  20,  20,  10,   0,   0,
     0,  10,   0,   0,   0,   0,  10,   0,
     0,  30,   0,   0,   0,   0,  30,   0,
     0,   0, -10,   0,   0, -10,   0,   0

};

// rook positional score
const int positionalRookScore[64] =
{
    50,  50,  50,  50,  50,  50,  50,  50,
    50,  50,  50,  50,  50,  50,  50,  50,
     0,   0,  10,  20,  20,  10,   0,   0,
     0,   0,  10,  20,  20,  10,   0,   0,
     0,   0,  10,  20,  20,  10,   0,   0,
     0,   0,  10,  20,  20,  10,   0,   0,
     0,   0,  10,  20,  20,  10,   0,   0,
     0,   0,   0,  20,  20,   0,   0,   0

};

// king positional score
const int positionalKingScore[64] = 
{
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   5,   5,   5,   5,   0,   0,
     0,   5,   5,  10,  10,   5,   5,   0,
     0,   5,  10,  20,  20,  10,   5,   0,
     0,   5,  10,  20,  20,  10,   5,   0,
     0,   0,   5,  10,  10,   5,   0,   0,
     0,   5,   5,  -5,  -5,   0,   5,   0,
     0,   0,   5,   0, -15,   0,  10,   0
};

// mirror positional score tables for opposite side
const int mirrorScope[128] =
{
	a1, b1, c1, d1, e1, f1, g1, h1,
	a2, b2, c2, d2, e2, f2, g2, h2,
	a3, b3, c3, d3, e3, f3, g3, h3,
	a4, b4, c4, d4, e4, f4, g4, h4,
	a5, b5, c5, d5, e5, f5, g5, h5,
	a6, b6, c6, d6, e6, f6, g6, h6,
	a7, b7, c7, d7, e7, f7, g7, h7,
	a8, b8, c8, d8, e8, f8, g8, h8
};

/*
    ADVANCED EVALUATION
    1) File Mask for a given square
    2) Rank Mask for a given square
    3) Isolated Pawn Mask for a given square
    4) Passed Pawn Maks for a given square
*/

// File Mask [square]
U64 fileMask[64];

// Rank Mask [square]
U64 rankMask[64];

// Isolated Pawn Mask [square] -> If No Pawn on left or right file then isolated
U64 isolatedPawnMask[64];

// White Passed Pawn Mask [square]
U64 whitePassedPawnMask[64];

// Black Passed Pawn Mask [square]
U64 blackPassedPawnMask[64];

// extract rank from a square [square]
const int getRankFromSquare[64] =
{
    7, 7, 7, 7, 7, 7, 7, 7,
    6, 6, 6, 6, 6, 6, 6, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0
};

// double pawns penalty
const int doublePawnPenalty = -10;

// isolated pawn penalty
const int isolatedPawnPenalty = -10;

// passed pawn bonus
const int passedPawnAdvantage[8] = { 0, 10, 30, 50, 75, 100, 150, 200 };

// Semi Open File Score
const int semiOpenFileAdvantage = 10;

// Open File Advantage Score
const int openFileAdvantage = 15;

// King Shield Bonus
const int kingShieldBonus = 5;

// Set File/Rank Mask for a given square
U64 setRankAndFileMask(int file , int rank){
    U64 mask = 0ULL;

    // Loop over all files and ranks
    for(int r = 0; r < 8; r++){
        for(int f = 0; f < 8; f++){
            // Init Square
            int square = 8 * r + f;

            if(file != -1){
                if(f == file) mask |= setBitOnSquare(mask , square);
            }
            else if(rank != -1){
                if(r == rank) mask |= setBitOnSquare(mask , square);
            }
        }
    }

    return mask;
}

// Init Evaluation related function
void initEvaluationMasks(){
    
    // Creating File and Rank Masks
    for(int rank = 0; rank < 8; rank++){
        for(int file = 0; file < 8; file++){
            // Init Square
            int square = 8 * rank + file;

            // Init File Mask
            fileMask[square] |= setRankAndFileMask(file , -1);

            // Init Rank Mask
            rankMask[square] |= setRankAndFileMask(-1 , rank);
        }
    }
    
    // Creating Isolated Pawn Masks
    for(int rank = 0; rank < 8; rank++){
        for(int file = 0; file < 8; file++){
            // Init Square
            int square = 8 * rank + file;

            // Init File Mask for the File on the Left of the square
            isolatedPawnMask[square] |= setRankAndFileMask(file - 1 , -1);

            // Init File Mask for the File on the Right of the square
            isolatedPawnMask[square] |= setRankAndFileMask(file + 1 , -1);

            // printf("%s\n" , squareToCoordinates[square]);
            // printBitBoard(isolatedPawnMask[square]);
        }
    }
    
    // Creating White Passed Pawn Masks
    for(int rank = 0; rank < 8; rank++){
        for(int file = 0; file < 8; file++){
            // Init Square
            int square = 8 * rank + file;

            // Init White Passed Pawn Mask for the File on the Left of the square
            whitePassedPawnMask[square] |= setRankAndFileMask(file - 1 , -1);

            // Init White Passed Pawn Mask for the File of the square
            whitePassedPawnMask[square] |= setRankAndFileMask(file , -1);

            // Init White Passed Pawn Mask for the File on the Right of the square
            whitePassedPawnMask[square] |= setRankAndFileMask(file + 1 , -1);

            // Remove redundant ranks from mask
            for(int r = 0; r < (8-rank); r++){
                whitePassedPawnMask[square] &= ~(rankMask[(8 *(7-r) + file)]);
            }

            // printf("%s\n" , squareToCoordinates[square]);
            // printBitBoard(whitePassedPawnMask[square]);
        }
    }
    
    // Creating Black Passed Pawn Masks
    for(int rank = 0; rank < 8; rank++){
        for(int file = 0; file < 8; file++){
            // Init Square
            int square = 8 * rank + file;

            // Init White Passed Pawn Mask for the File on the Left of the square
            blackPassedPawnMask[square] |= setRankAndFileMask(file - 1 , -1);

            // Init White Passed Pawn Mask for the File of the square
            blackPassedPawnMask[square] |= setRankAndFileMask(file , -1);

            // Init White Passed Pawn Mask for the File on the Right of the square
            blackPassedPawnMask[square] |= setRankAndFileMask(file + 1 , -1);

            // Remove redundant ranks from mask
            for(int r = 0; r < (rank + 1); r++){
                blackPassedPawnMask[square] &= ~(rankMask[(8 * r + file)]);
            }

            // printf("%s\n" , squareToCoordinates[square]);
            // printBitBoard(blackPassedPawnMask[square]);
        }
    }
}

/*
    TAPERED EVALUATION
*/

// material score [game phase][piece]
const int materialScore[2][12] =
{
    // opening material score
    82, 337, 365, 477, 1025, 12000, -82, -337, -365, -477, -1025, -12000,
    
    // endgame material score
    94, 281, 297, 512,  936, 12000, -94, -281, -297, -512,  -936, -12000
};

// game phase scores
const int openingPhaseScore = 6192;
const int endgamePhaseScore = 518;

// game phases
enum { OPENING, ENDGAME, MIDDLEGAME };

// piece types
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

// positional piece scores [game phase][piece][square]
const int positionalScore[2][6][64] =
// 2 phases : OPeing & endgame , middlegame is found by interplolating b/w the 2

// opening positional piece scores //
{
    //pawn
    0,   0,   0,   0,   0,   0,  0,   0,
    98, 134,  61,  95,  68, 126, 34, -11,
    -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
    0,   0,   0,   0,   0,   0,  0,   0,
    
    // knight
    -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
    -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
    
    // bishop
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -4,   5,  19,  50,  37,  37,   7,  -2,
    -6,  13,  13,  26,  34,  12,  10,   4,
    0,  15,  15,  15,  14,  27,  18,  10,
    4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
    
    // rook
    32,  42,  32,  51, 63,  9,  31,  43,
    27,  32,  58,  62, 80, 67,  26,  44,
    -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
    
    // queen
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -1, -18,  -9,  10, -15, -25, -31, -50,
    
    // king
    -65,  23,  16, -15, -56, -34,   2,  13,
    29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
    1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,


    // Endgame positional piece scores //

    //pawn
    0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
    94, 100,  85,  67,  56,  53,  82,  84,
    32,  24,  13,   5,  -2,   4,  17,  17,
    13,   9,  -3,  -7,  -7,  -8,   3,  -1,
    4,   7,  -6,   1,   0,  -5,  -1,  -8,
    13,   8,   8,  10,  13,   0,   2,  -7,
    0,   0,   0,   0,   0,   0,   0,   0,
    
    // knight
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
    
    // bishop
    -14, -21, -11,  -8, -7,  -9, -17, -24,
    -8,  -4,   7, -12, -3, -13,  -4, -14,
    2,  -8,   0,  -1, -2,   6,   0,   4,
    -3,   9,  12,   9, 14,  10,   3,   2,
    -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
    
    // rook
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
    7,  7,  7,  5,  4,  -3,  -5,  -3,
    4,  3, 13,  1,  2,   1,  -1,   2,
    3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
    
    // queen
    -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
    3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
    
    // king
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
    10,  17,  23,  15,  20,  45,  44,  13,
    -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
};

static inline int getGamePhaseScore(){
    /*
        Derived from the pieces (not counting pawns and kings) that are still on the board.
        Full Material starting position game hase score is defined as : 
        4 * Knights Material Score in Opening + 
        4 * Bishop Material Score in Opening + 
        4 * Rook Material Score in Opening + 
        2 * Queen Material Score in Opening + 
    */

    // Phase Scores
    int whitePiecesScore = 0 , blackPiecesScore = 0;

    // Calculate Number of Pieces 
    for(int piece = N; piece <=Q; piece++){
        whitePiecesScore += countBits(pieceBitBoards[piece]) * materialScore[OPENING][piece];
    }
    for(int piece = n; piece <=q; piece++){
        blackPiecesScore += countBits(pieceBitBoards[piece]) * materialScore[OPENING][piece];
    }

    return whitePiecesScore + blackPiecesScore;
}

static inline int evaluate(){

    // Get Game Phase Score
    int gamePhaseScore = getGamePhaseScore();

    // Current Game Phase
    int gamePhase = -1; // Neither Opening , Endgame or Middlegame

    // Get Current Game Phase value based on Game Phase Score
    if(gamePhaseScore > openingPhaseScore){
        gamePhase = OPENING;
    } else if(gamePhaseScore < endgamePhaseScore){
        gamePhase = ENDGAME;
    } else {
        gamePhase = MIDDLEGAME;
    }
    
    // Static Evaluation Square
    int score = 0;

    // Current Piece BitBoard Copy
    U64 bitBoardOfPiece;

    // Track number of pawns in a file
    int pawnsInFileCount = 0;

    // Init Piece & Square
    int piece , square;

    for(int consideredPiece = P; consideredPiece <= k; consideredPiece++){
        
        //Init Piece BitBoard Copy
        bitBoardOfPiece = pieceBitBoards[consideredPiece];

        while(bitBoardOfPiece){

            // Init Piece
            piece = consideredPiece;
            
            // Init Square
            square = getLS1BIndex(bitBoardOfPiece);

            if (gamePhase == MIDDLEGAME){
                // If Middlegame interpolate the points to get the score
                // Formula : 
                // Middlgame Score = ((Piece_Score_Opening * Gamephase_Score) + ((Piece_Score_Endgame) * (Opening_Score_Const - Gamephase_Score))) / Opening_Score_Const
                //  E.g. the score for pawn on d4 at phase say 5000 would be
                // interpolated_score = (12 * 5000 + ((-7) * (6192 - 5000))) / 6192 = 8,342377261

                score += (
                    (
                        (   materialScore[OPENING][piece] * 
                            gamePhaseScore
                        ) +
                        (
                            materialScore[ENDGAME][piece] * 
                            (openingPhaseScore - gamePhaseScore)
                        )
                    ) / openingPhaseScore
                );
            }
            else{
                // Score based on material for opening and endgame cause the points are adjusted for it
                score += materialScore[gamePhase][piece];
            }

            // score positional piece scores
            switch (piece)
            {
                // evaluate white pieces
                case P: 

                    if (gamePhase == MIDDLEGAME){

                        score += (
                            (
                                (   positionalScore[OPENING][PAWN][square] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][PAWN][square] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score += positionalScore[gamePhase][PAWN][square];
                    }
                    
                    // // Add Positional Score
                    // score += positionalPawnScore[square]; 

                    // // Count the number of double pawns White has
                    // pawnsInFileCount = countBits(pieceBitBoards[P] & fileMask[square]);

                    // // If Number of Pawns in a file > 1 => double, triple pawns exsist
                    // if(pawnsInFileCount > 1){
                    //     score += pawnsInFileCount * doublePawnPenalty;
                    // }

                    // // Check Presence of Isolated Pawns
                    // if ((pieceBitBoards[P] & isolatedPawnMask[square]) == 0){
                    //     score += isolatedPawnPenalty;
                    // }

                    // // Check Presence of White Passed Pawns
                    // if ((whitePassedPawnMask[square] & pieceBitBoards[p]) == 0){
                    //     score += passedPawnAdvantage[getRankFromSquare[square]];
                    // }

                    break;
                case N: 
                    // score += positionalKnightScore[square]; 

                    if (gamePhase == MIDDLEGAME){

                        score += (
                            (
                                (   positionalScore[OPENING][KNIGHT][square] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][KNIGHT][square] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score += positionalScore[gamePhase][KNIGHT][square];
                    }

                    break;
                case B:  
                    // // Bishop Positional Score
                    // score += positionalBishopScore[square]; 

                    // // Mobility Advantage
                    // score += countBits(getBishopAttacks(square , boardOccupancies[BOTH]));

                    if (gamePhase == MIDDLEGAME){

                        score += (
                            (
                                (   positionalScore[OPENING][BISHOP][square] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][BISHOP][square] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score += positionalScore[gamePhase][BISHOP][square];
                    }

                    break;
                case Q:   

                    // Mobility Advantage
                    // score += countBits(getQueenAttacks(square , boardOccupancies[BOTH]));

                    if (gamePhase == MIDDLEGAME){

                        score += (
                            (
                                (   positionalScore[OPENING][QUEEN][square] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][QUEEN][square] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score += positionalScore[gamePhase][QUEEN][square];
                    }

                    break;
                case R: 
                    // // Rook Positional Score
                    // score += positionalRookScore[square]; 

                    // // Open File Bonus
                    // if(((pieceBitBoards[P] | pieceBitBoards[p]) & fileMask[square]) == 0) score += openFileAdvantage;
                    // // Semi Open File Bonus
                    // if((pieceBitBoards[P] & fileMask[square]) == 0) score += semiOpenFileAdvantage;

                    if (gamePhase == MIDDLEGAME){

                        score += (
                            (
                                (   positionalScore[OPENING][ROOK][square] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][ROOK][square] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score += positionalScore[gamePhase][ROOK][square];
                    }

                    break;
                case K: 
                    // // King Positional Score
                    // score += positionalKingScore[square]; 

                    // // Open File Penalty
                    // if(((pieceBitBoards[P] | pieceBitBoards[p]) & fileMask[square]) == 0) score -= openFileAdvantage;
                    // // Semi Open File Penalty
                    // if((pieceBitBoards[P] & fileMask[square]) == 0) score -= semiOpenFileAdvantage;

                    // // King Shield Bonus
                    // score += countBits(kingAttacks[square] & boardOccupancies[WHITE]) * kingShieldBonus;

                    if (gamePhase == MIDDLEGAME){

                        score += (
                            (
                                (   positionalScore[OPENING][KING][square] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][KING][square] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score += positionalScore[gamePhase][KING][square];
                    }

                    break;

                // evaluate black pieces
                case p: 
                    if (gamePhase == MIDDLEGAME){

                        score -= (
                            (
                                (   positionalScore[OPENING][PAWN][mirrorScope[square]] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][PAWN][mirrorScope[square]] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score -= positionalScore[gamePhase][PAWN][mirrorScope[square]];
                    }
                                        
                    // // Add Positional Score
                    // score -= positionalPawnScore[mirrorScope[square]]; 

                    // // Count the number of double pawns White has
                    // pawnsInFileCount = countBits(pieceBitBoards[p] & fileMask[square]);

                    // // If Number of Pawns in a file > 1 => double, triple pawns exsist
                    // if(pawnsInFileCount > 1){
                    //     score -= pawnsInFileCount * doublePawnPenalty;
                    // }

                    // // Check Presence of Isolated Pawns
                    // if ((pieceBitBoards[p] & isolatedPawnMask[square]) == 0){
                    //     score -= isolatedPawnPenalty;
                    // }
                    
                    // // Check Presence of Black Passed Pawns
                    // if ((blackPassedPawnMask[square] & pieceBitBoards[P]) == 0){
                    //     score -= passedPawnAdvantage[getRankFromSquare[mirrorScope[square]]];
                    // }
                    break;
                case n: 
                    // score -= positionalKnightScore[mirrorScope[square]]; 

                    if (gamePhase == MIDDLEGAME){

                        score -= (
                            (
                                (   positionalScore[OPENING][KNIGHT][mirrorScope[square]] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][KNIGHT][mirrorScope[square]] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score -= positionalScore[gamePhase][KNIGHT][mirrorScope[square]];
                    }
                    break;
                case b: 
                    // // Bishop Positional Score
                    // score -= positionalBishopScore[mirrorScope[square]]; 

                    // // Mobility Advantage
                    // score -= countBits(getBishopAttacks(square , boardOccupancies[BOTH]));

                    if (gamePhase == MIDDLEGAME){

                        score -= (
                            (
                                (   positionalScore[OPENING][BISHOP][mirrorScope[square]] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][BISHOP][mirrorScope[square]] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score -= positionalScore[gamePhase][BISHOP][mirrorScope[square]];
                    }

                    break;
                case q: 

                    // Mobility Advantage
                    // score -= countBits(getQueenAttacks(square , boardOccupancies[BOTH]));

                    if (gamePhase == MIDDLEGAME){

                        score -= (
                            (
                                (   positionalScore[OPENING][QUEEN][mirrorScope[square]] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][QUEEN][mirrorScope[square]] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score -= positionalScore[gamePhase][QUEEN][mirrorScope[square]];
                    }

                    break;
                case r: 
                    // // Rook Positional Score
                    // score -= positionalRookScore[mirrorScope[square]];

                    // // Open File Bonus
                    // if(((pieceBitBoards[p] | pieceBitBoards[P]) & fileMask[square]) == 0) score -= openFileAdvantage;
                    // // Semi Open File Bonus
                    // if((pieceBitBoards[p] & fileMask[square]) == 0) score -= semiOpenFileAdvantage;

                    if (gamePhase == MIDDLEGAME){

                        score -= (
                            (
                                (   positionalScore[OPENING][ROOK][mirrorScope[square]] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][ROOK][mirrorScope[square]] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score -= positionalScore[gamePhase][ROOK][mirrorScope[square]];
                    }
                    break;
                case k: 
                    // // King Positional Score
                    // score -= positionalKingScore[mirrorScope[square]]; 

                    // // Open File Penalty
                    // if(((pieceBitBoards[p] | pieceBitBoards[P]) & fileMask[square]) == 0) score += openFileAdvantage;
                    // // Semi Open File Penalty
                    // if((pieceBitBoards[p] & fileMask[square]) == 0) score += semiOpenFileAdvantage;

                    // // King Shield Bonus
                    // score -= countBits(kingAttacks[square] & boardOccupancies[BLACK]) * kingShieldBonus;

                    if (gamePhase == MIDDLEGAME){

                        score -= (
                            (
                                (   positionalScore[OPENING][KING][mirrorScope[square]] * 
                                    gamePhaseScore
                                ) +
                                (
                                    positionalScore[ENDGAME][KING][mirrorScope[square]] * 
                                    (openingPhaseScore - gamePhaseScore)
                                )
                            ) / openingPhaseScore
                        );
                    }
                    else{
                        // Score based on material for opening and endgame cause the points are adjusted for it
                        score -= positionalScore[gamePhase][KING][mirrorScope[square]];
                    }
                    
                    break;
            }

            // Pop MS1B
            popBitOnSquare(bitBoardOfPiece , square);
        }

    }
    return (sideToMove == WHITE) ? score : (-score); // For NegaMax Requirements
}

/*
    MOVE ORDERING
    For the alpha-beta algorithm to perform well, the best moves need to be searched first (need to be kept on the left side of the tree)
    1) PV Move
    2) Captures in MVV-LVA
    3) 1st Killer Move
    4) 2nd Killer Move
    5) History Moves
    6) Unsorted Moves
*/

// MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
//  This technique assumes that the best capture will be the one that captures the biggest piece.
//  If more than one of your pieces can capture a big piece, the assumption is made that it is best to capture with the smallest piece.
// The advantage of MVV/LVA is that this is easy to implement and it results in a high nodes/second.  
// The disadvantage is that your search is inefficient -- you spend most of your time evaluating losing captures, so you search less deeply.

/*
                          
    (Victims) Pawn Knight Bishop   Rook  Queen   King
  (Attackers)
        Pawn   105    205    305    405    505    605
      Knight   104    204    304    404    504    604
      Bishop   103    203    303    403    503    603
        Rook   102    202    302    402    502    602
       Queen   101    201    301    401    501    601
        King   100    200    300    400    500    600

*/

#define MAX_PLY 64

// MVV LVA [attacker][victim]
static int MVV_LVA[12][12] = {
 	105, 205, 305, 405, 505, 605,  105, 205, 305, 405, 505, 605,
	104, 204, 304, 404, 504, 604,  104, 204, 304, 404, 504, 604,
	103, 203, 303, 403, 503, 603,  103, 203, 303, 403, 503, 603,
	102, 202, 302, 402, 502, 602,  102, 202, 302, 402, 502, 602,
	101, 201, 301, 401, 501, 601,  101, 201, 301, 401, 501, 601,
	100, 200, 300, 400, 500, 600,  100, 200, 300, 400, 500, 600,

	105, 205, 305, 405, 505, 605,  105, 205, 305, 405, 505, 605,
	104, 204, 304, 404, 504, 604,  104, 204, 304, 404, 504, 604,
	103, 203, 303, 403, 503, 603,  103, 203, 303, 403, 503, 603,
	102, 202, 302, 402, 502, 602,  102, 202, 302, 402, 502, 602,
	101, 201, 301, 401, 501, 601,  101, 201, 301, 401, 501, 601,
	100, 200, 300, 400, 500, 600,  100, 200, 300, 400, 500, 600
};

// Killer moves work on the supposition that most of the moves do not change the situation on the board too much. 
// For example if a program decides that expelling a black bishop from b4 by a move a2-a3 is good
// then it is likely to work whatever Black played on the previous move
// Killer Moves[id][ply]
static int killerMoves[2][MAX_PLY];

// a dynamic move ordering method based on the number of cutoffs caused by a given move irrespectively from the position in which the move has been made. 
// history heuristic is often presented as depth-independent generalization of the killer moves
// History Moves[piece][square]
static int historyMoves[12][64];

/*
      ================================
            Triangular PV table
      --------------------------------
        PV line: e2e4 e7e5 g1f3 b8c6
      ================================

           0    1    2    3    4    5
      
      0    m1   m2   m3   m4   m5   m6
      
      1    0    m2   m3   m4   m5   m6 
      
      2    0    0    m3   m4   m5   m6
      
      3    0    0    0    m4   m5   m6
       
      4    0    0    0    0    m5   m6
      
      5    0    0    0    0    0    m6
*/

// PV length [ply]
int pvLength[MAX_PLY];

// PV table [ply] [ply]
int pvTable[MAX_PLY][MAX_PLY];

void printPVTable(){
    int emptyFlag = 0;
    printf("\t \t PV Table : \n");
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            if(pvTable[i][j] != 0){
                printMove(pvTable[i][j]);
                printf(" ");
            }
            else{
                break;
            }
        }
        printf("\n");
    }
}

// Flags to determine if we are following the PV from the PV table and the score of the PV
int followPV , scorePV;

/*
    TRANSPOSITION TABLE
*/

U64 transpositionTableEntries = 0;

#define HASH_NOT_FOUND_EVAL 100000 // This value should be outside [alpha , beta]

// transposition Hash Flags
#define hashPV 0 // This is for hashing PV Nodes
#define hashALPHA 1 // This is for fail low moves
#define hashBETA  2 // This is for fail high moves

// Transposition table Entry Data Structure
typedef struct { // Each Entry is 20B
    U64 key; // Hash for the position
    int depth; // Current depth during search
    int flags; // Tag the type of Node (Fail-High(score >= beta) , Fail-Low(score <=alpha) , PV Node(score > alpha && score < beta))
    int eval; // Score (alpha/beta/PV)
} TranspositionTable; // aka Hash Table

// Define the Transposition table(TT) store
TranspositionTable * transpositionTable = NULL;

// Clear the TT - > Not done by default for structs by compiler

void clearTranspositionTable(){

    // Init Hash Table Entry Ptr
    TranspositionTable * hashEntry;

    for(hashEntry = transpositionTable; hashEntry < transpositionTable + transpositionTableEntries; hashEntry++){
        // Reset the TT fields to 0
        hashEntry->key = 0;
        hashEntry->depth = 0;
        hashEntry->flags = 0;
        hashEntry->eval = 0;
    }
}

void initTranspositionTable(int sizeInMB){
    // Hash Table Size
    int hashTableSize = 0x100000 * sizeInMB;

    // Init Number of Hash Entries
    transpositionTableEntries = hashTableSize / sizeof(TranspositionTable);

    // Free hash Table if not empty
    if(transpositionTable != NULL) {
        printf("    Clearing hash memory...\n");
        free(transpositionTable);
    }

    // Allocate Memory
    transpositionTable = (TranspositionTable *)malloc(sizeof(TranspositionTable) * transpositionTableEntries);

    // If Allocation Failed , try with half size
    if(transpositionTable == NULL){

        printf("    Couldn't allocate memory for hash table, tryinr %dMB...", sizeInMB / 2);
        initTranspositionTable(sizeInMB / 2);
    } 
    // If Allocated , clear the TT for entires
    else{
        clearTranspositionTable();
        printf("    Hash table is initialied with %llu entries\n", transpositionTableEntries);
    } 
}
// Read Values from Hash Table

static inline int readFromHashTable(int alpha , int beta , int depth){
    // Assign a pointer to where the current hash/position at the current depth should be in the table
    TranspositionTable * ttPtr = &transpositionTable[hashKey % transpositionTableEntries]; // Ptr to make in-place updates

    // Check if we have the same position/hash from the referenced position in the TT
    if(ttPtr->key == hashKey){
        // We have the dsame position/hash
        if(ttPtr->depth >= depth){ 
            // If depth >= ,than it has already been analysed much better or equal to curr depth so its eval is better.

            int eval = ttPtr->eval;
            if (eval < (-MATE_SCORE)) eval -= ply;
            if (eval > MATE_SCORE) eval += ply;

            // If node is of type PV Node
            if(ttPtr->flags == hashPV){
                // printf("Exact Score : \n");
                return eval;
            }
            else if( (ttPtr->flags == hashALPHA) && (eval <= alpha) ){
                // printf("Alpha Score : \n");
                return alpha;
            }
            else if( (ttPtr->flags == hashBETA) && (eval >= beta) ){
                // printf("Beta Score : \n");
                return beta;
            }
        }
    }
    // If hash is not same/not found
    // printf("Not Found Score Score : \n");
    return HASH_NOT_FOUND_EVAL;

}

static inline void writeToHashTable(int evaluation , int depth , int hashFlag){
    // Assign a pointer to where the current hash/position at the current depth should be in the table
    TranspositionTable * ttPtr = &transpositionTable[hashKey % transpositionTableEntries]; // Ptr to make in-place updates
    if (evaluation < (-MATE_SCORE)) evaluation -= ply;
    if (evaluation > MATE_SCORE) evaluation += ply;
    ttPtr->key = hashKey;
    ttPtr->depth = depth;
    ttPtr->eval = evaluation;
    ttPtr->flags = hashFlag;
}

/*
    PRINCIPLE VARIATION Implementation
*/

static inline void enablePVScoring(Move * moveList , int depth){

    // Disable Follow PV
    followPV = 0;
    for(int count = 0; count < moveList->moveCount; count++){
        if(pvTable[0][ply] == moveList->moves[count]){
            // Enable Pv Scoring
            scorePV = 1;

            // Enable Follow PV
            followPV = 1;
        }
    }
}

static inline int scoreMove(int move){
    // Has 3 type 1) PV Moves , 2) Captures , 3) Quite

    // If PV Scoring is enabled
    if(scorePV){
        if(pvTable[0][ply] == move){

            // Disable score PV
            scorePV = 0;

            // Give highest score to the PV Move to search it 1st
            return 20000; // 2* capture (Random)
        }
    }


    if(extractCaptureFlag(move)){
        // Captures

        //init Target Piece
        int targetPiece = P;

        // Defines what ranges to iterate over , start to end
        int startPiece , endPiece; 

        if(sideToMove == WHITE){
            startPiece = p;
            endPiece = k;
        }
        else{
            startPiece = P;
            endPiece = K;
        }
        
        // Loop Over the opponents(other side to move)'s bitboard 
        for(int consideredPiece = startPiece ; consideredPiece <= endPiece ; consideredPiece++){
            // If the current piece type is present on the target square
            if(getBitOnSquare(pieceBitBoards[consideredPiece] , extractTarget(move))){
                // Remove the corresponding piece from the target square 
                targetPiece = consideredPiece;
                break; // only 1 piece possible for a square (acc to rules) => end the search
            }                
        }
        // printMove(move);
        // printf("\n");
        // printf("Src Piece: %c\n" , asciiPieces[extractMovedPiece(move)]);
        // printf("Target Piece: %c\n" , asciiPieces[targetPiece]);

        // Score based on MVV_LVA[srcPiece][targetPiece]
        return MVV_LVA[extractMovedPiece(move)][targetPiece] + 10000; // To give it a higher weightage than killer moves which start at 9000
    }
    else{
        // Quite Move

        // Score 1st killer Move
        if(killerMoves[0][ply] == move){
            return 9000;
        }
        // Score 2nd killer Move
        else if(killerMoves[1][ply] == move){
            return 8000;
        }
        // Score History Move
        else{
            return historyMoves[extractMovedPiece(move)][extractTarget(move)];
        }

    }

    return 0;
}

void printMoveScores(Move * moveList){
    printf("      Move Scores : \n");
    for(int count = 0; count < moveList->moveCount; count++){
        printf("   Move : ");
        printMove(moveList->moves[count]);
        printf(" Score : %d\n" , scoreMove(moveList->moves[count]));
    }
}

//Sort moves in descending order of score for search
static inline int sortMoves(Move * moveList){
    int moveScores[moveList->moveCount];

    for(int count = 0; count < moveList->moveCount;count++){
        moveScores[count] = scoreMove(moveList->moves[count]);
    }
    for(int current = 0; current < moveList->moveCount; current++){
        for(int next = current+1; next < moveList->moveCount; next++){
            if(moveScores[current] < moveScores[next]){
                // Swap Scores
                int temp = moveScores[next];
                moveScores[next] = moveScores[current];
                moveScores[current] = temp;

                // Swap Moves
                int tempMove = moveList->moves[next];
                moveList->moves[next] = moveList->moves[current];
                moveList->moves[current] = tempMove;
            }
        }
    }

}

/*
    Time Control Flags Required for UCI 
    kept above Search since it is used there to stop search based on cmd
*/

// Exit from engine flag
int quit = 0;

// UCI "movestogo" cmd move counter
int movesToGo = 40;

// UCI "movetime" cmd time counter
int moveTime = -1;

// UCI "time" cmd holder (ms)
int time = -1;

// UCI "inc" cmd's time increment holder
int inc = 0;

// UCI "starttime" cmd time holder
int startTime = 0;

// UCI "stopttime" cmd time holder
int stopTime = 0;

// variable to flag time control availability
int timeSet = 0;

// variable to flag when time is up
int stopped = 0;

/*

  Function to "listen" to GUI's input during search.
  It's waiting for the user input from STDIN.
  OS dependent.
  
  First Richard Allbert aka BluefeverSoftware grabbed it from somewhere...
  And then Code Monkey King has grabbed it from VICE)
  
*/

int getTimeInMilliSeconds(){
    #ifdef WIN64
        return GetTickCount();
    #else 
        struct timeval timeValue;
        gettimeofday(&timeValue , NULL);
        return timeValue.tv_sec * 1000 + timeValue.tv_usec / 1000; 
    #endif 
}
  
int inputWaiting()
{
    #ifndef WIN32
        fd_set readfds;
        struct timeval tv;
        FD_ZERO (&readfds);
        FD_SET (fileno(stdin), &readfds);
        tv.tv_sec=0; tv.tv_usec=0;
        select(16, &readfds, 0, 0, &tv);

        return (FD_ISSET(fileno(stdin), &readfds));
    #else
        static int init = 0, pipe;
        static HANDLE inh;
        DWORD dw;

        if (!init)
        {
            init = 1;
            inh = GetStdHandle(STD_INPUT_HANDLE);
            pipe = !GetConsoleMode(inh, &dw);
            if (!pipe)
            {
                SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT|ENABLE_WINDOW_INPUT));
                FlushConsoleInputBuffer(inh);
            }
        }
        
        if (pipe)
        {
           if (!PeekNamedPipe(inh, NULL, 0, NULL, &dw, NULL)) return 1;
           return dw;
        }
        
        else
        {
           GetNumberOfConsoleInputEvents(inh, &dw);
           return dw <= 1 ? 0 : dw;
        }

    #endif
}

// read GUI/user input
void readInput()
{
    // bytes to read holder
    int bytes;
    
    // GUI/user input
    char input[256] = "", *endc;

    // "listen" to STDIN
    if (inputWaiting())
    {
        // tell engine to stop calculating
        stopped = 1;
        
        // loop to read bytes from STDIN
        do
        {
            // read bytes from STDIN
            bytes=read(fileno(stdin), input, 256);
        }
        
        // until bytes available
        while (bytes < 0);
        
        // searches for the first occurrence of '\n'
        endc = strchr(input,'\n');
        
        // if found new line set value at pointer to 0
        if (endc) *endc=0;
        
        // if input is available
        if (strlen(input) > 0)
        {
            // match UCI "quit" command
            if (!strncmp(input, "quit", 4))
            {
                // tell engine to terminate exacution    
                quit = 1;
            }

            // // match UCI "stop" command
            else if (!strncmp(input, "stop", 4))    {
                // tell engine to terminate exacution
                quit = 1;
            }
        }   
    }
}

// a bridge function to interact between search and GUI input
static void communicate() {
	// if time is up break here
    if(timeSet == 1 && (getTimeInMilliSeconds() > stopTime)) {
		// tell engine to stop calculating
		stopped = 1;
	}
	
    // read GUI input
	readInput();
}


/*
    SEARCH

    Ver. 0.1 :  Vanilla Negamax Alpha Beta Pruning Search
        alpha initially is -inf and 1 side keeps trying to increase it 
        beta initially is inf and other side keeps trying to decrease it 

        Fail-Hard Approach : Score cannot go beyond the [alpha , beta] range -> Easier to implement , if below beta => Fail-Hard , //ly for alpha => Fail Low
        Fail-Soft Approach : Score can go beyond the [alpha , beta] range

    Ver. 0.5 : Quiescence Search + Negamax 
        Most chess programs, at the end of the main search perform a more limited quiescence search, containing fewer moves. 
        The purpose of this search is to only evaluate "quiet" positions, or positions where there are no winning tactical moves to be made. 
        This search is needed to avoid the horizon effect. Simply stopping your search when you reach the desired depth and then evaluate, is very dangerous. 
        Consider the situation where the last move you consider is QxP. If you stop there and evaluate, you might think that you have won a pawn. 
        But what if you were to search one move deeper and find that the next move is PxQ? You didn't win a pawn, you actually lost a queen. 
        Hence the need to make sure that you are evaluating only quiescent (quiet) positions.
    
    Ver. 1.0 : PVS + Null Move Pruning + LMR
*/

// If score belongs to [mate_score , mate_value] => Write Mate Score.

U64 searchNodes;

// Best Move (Will be redundant once PV is introduced though)
// int bestMove;

static inline int isRepeated(){
    // Loop over the repetition indx range
    for(int idx = 0; idx < repetitionIndex; idx++){
        // If We find a hash key with the same hash key as current position
        if(repetitionTable[idx] == hashKey){
            return 1;
        }
    }
    
    // If No repetition Found 
    return 0;
}

//Quiescence Search

static inline int quiescenceSearch(int alpha , int beta){

    if((searchNodes & 2047) == 0){
        // "listen" to the GUI/User Input -> to check if time up or game ended
        communicate();
    }

    searchNodes++;

    //Recursive Base Case
    
    // If too deep => Can cause overflow so return
    if (ply > (MAX_PLY - 1)) return evaluate();

    int evaluation = evaluate();

    // Fail-Hard Beta Cutoff
    if (evaluation >= beta) return beta; // Fail High Case => Found Worse Move

    if (evaluation > alpha){ // Found a better move (defined as) PV Node
        
        // Keep this as the current best score and discard scores worse than this.
        alpha = evaluation;        
    } 
    // create a MoveList Instance
    Move moveList[1];

    // generate Moves
    generateMoves(moveList);

    // Sort Moves
    sortMoves(moveList);

    // Loop over the moves in moveList
    for(int count = 0; count < moveList->moveCount; count++){

        // preserve board state
        copyBoardState();

        // increment ply
        ply++;

        // Increment Repetition index & Store Hash Key
        repetitionIndex++;
        repetitionTable[repetitionIndex] = hashKey;

        // Make only legal moves
        if(makeMove(moveList->moves[count] , captureMoves) == 0){
            // If Move is Illegal

            // Decrement counter since we are gonna revert back
            ply--;

            // Decrement Repetition Index
            repetitionIndex--;

            //Skip to Next Move
            continue;
        }

        // Score the current Move
        int score = -quiescenceSearch(-beta , -alpha);

        // Decrement counter since we are gonna revert back
        ply--;

        // Decrement Repetition Index
        repetitionIndex--;

        // takeback Move the Legal move
        restoreBoardState();

        // return 0 if time is up
        if(stopped == 1) return 0;

        if (score > alpha){ // Found a better move (defined as) PV Node
            
            // Keep this as the current best score and discard scores worse than this.
            alpha = score;   

            // fail-hard beta cutoff
            if (score >= beta)
            {
                // node (position) fails high
                return beta;
            }     
        } 
    }

    // Node Fails Low Case
    return alpha;
}

// Negamax Alpha Beta Search

const int fullDepthMoves = 4;
const int reductionLimit = 3;

// depth = maximum ply allowed in search before evaluating the final/horizon position.
static inline int negaMaxSearch(int alpha , int beta , int depth){

    // Define score to store the evaluation of a move -> Static Evaluation perspective
    int score;

    int hashFlag = hashALPHA; // Assume is is the bad move

    // If Position Repeats , return Draw score
    if (ply && isRepeated()) return 0;

    // To figure out if a node is PV Node or not
    int pvNode = (beta - alpha > 1);

    // Read Hash Entry if node not a root Node and hash entry is available and current node is not a PV Node
    if(ply && ((score = readFromHashTable(alpha , beta , depth)) != HASH_NOT_FOUND_EVAL) && (pvNode == 0)){
        // Already in TT , no need to redo the search for that move
        return score;
    }
    
    // Init PV Length
    pvLength[ply] = ply;

    if((searchNodes & 2047) == 0){
        // "listen" to the GUI/User Input -> to check if time up or game ended
        communicate();
    }

    // Recursion base Condition
    if (depth == 0){
        // return evaluate(); // For Vanilla Negamax

        // Quiescence Search
        return quiescenceSearch(alpha , beta);
    }

    // If too deep => Can cause overflow so return
    if (ply > (MAX_PLY - 1)) return evaluate();
    
    // Increment Nodes Count
    searchNodes++; // For later improvements to reduce search space -> as a performance measure

    // Chek If King in check
    int isCheck = isSquareAttacked((sideToMove == WHITE) ? getLS1BIndex(pieceBitBoards[K]) : getLS1BIndex(pieceBitBoards[k]) , sideToMove ^ 1);

    if(isCheck) depth++;

    // Legal Moves Counter
    int legalMoves = 0;

    // Null Moves Pruning
    if ((depth >= 3) && (isCheck == 0) && (ply)){
        
        // Preserve Booard State 
        copyBoardState();

        // Increment ply cause gonna give a free move to opponent 
        ply++;

        // Increment Repetition index & Store Hash Key
        repetitionIndex++;
        repetitionTable[repetitionIndex] = hashKey;
        
        if(enPassant != NULL_SQUARE){
            // Remove EnPassant Rights
            hashKey ^= enPassantKeys[enPassant];
        }

        // Reset enPassant Square
        enPassant = NULL_SQUARE;

        // Switch side -> literally give opponent a free move
        sideToMove ^= 1;
        
        // Hash the side now
        hashKey ^= sideToMoveKey;

        // depth - 1 - R is a reduction of depth , R = reduction limit
        score = -negaMaxSearch(-beta , -beta + 1 , depth - 1 - 2); // R = 2 here

        // Decrement ply cause gonna take move back
        ply--;

        // Decrement Repetition Index
        repetitionIndex--;

        // Restore Board State
        restoreBoardState();

        // Return 0 if time up
        if (stopped == 1) return 0;

        // Fail-Hard Cutoff
        if(score >= beta) return beta;

    }

    // create a MoveList Instance
    Move moveList[1];

    // generate Moves
    generateMoves(moveList);

    if(followPV) enablePVScoring(moveList , depth);
    
    // Sort the Move
    sortMoves(moveList);
    
    // Init No. of moves seahced
    int movesSearched = 0;

    // Loop over the moves in moveList
    for(int count = 0; count < moveList->moveCount; count++){

        // preserve board state
        copyBoardState();

        // increment ply
        ply++;

        // Increment Repetition index & Store Hash Key
        repetitionIndex++;
        repetitionTable[repetitionIndex] = hashKey;

        // Make only legal moves
        if(makeMove(moveList->moves[count] , allMoves) == 0){
            // If Move is Illegal

            // Decrement counter since we are gonna revert back
            ply--;

            // Decrement Repetition Index
            repetitionIndex--;

            //Skip to Next Move
            continue;
        }

        // Increment Legal Moves
        legalMoves++;


        // For all non-PV Node -> 1st Legal Move
        if(movesSearched == 0){
            // Score the current Move
            score = -negaMaxSearch(-beta , -alpha , depth-1);
        }
        else{
            // Condition to consider Late Move Reduction(LMR) -> Unsorted moves in ordering require lower depth consideration cause they are usually not worth it.
            if(
                (movesSearched >= fullDepthMoves) 
                && (depth >= reductionLimit) 
                && (isCheck == 0)
                && (extractCaptureFlag(moveList->moves[count]) == 0)
                && (extractPromotedPiece(moveList->moves[count]) == 0)
                ){
                // Search with reduced depth
                score = -negaMaxSearch(-alpha - 1 , -alpha , depth - 2 );
            }
            else{
                score = alpha + 1; // Hack to ensure the full depth search for the move
            }
            // PVS
            if(score > alpha){
                // Found a better move during LMR -> Use Principle Variation Seacrh (PVS)
                score = -negaMaxSearch(-alpha - 1 , -alpha , depth - 1 );
                if((score > alpha) && (score < beta)){
                    // LMR fails so use normal search
                    score = -negaMaxSearch(-beta , -alpha , depth - 1 );
                }
            }
        }
        // Decrement counter since we are gonna revert back
        ply--;

        // Decrement Repetition Index
        repetitionIndex--;

        // takeback Move the Legal move
        restoreBoardState();

        // Return 0 if time up
        if (stopped == 1) return 0;

        // Increment Moves Searched
        movesSearched++;

        if (score > alpha){ // Found a better move 

            // Since it is a PV Node
            hashFlag = hashPV;

            // Store move as History Move -> defined only for quite move
            if(extractCaptureFlag(moveList->moves[count]) == 0){
                historyMoves[extractMovedPiece(moveList->moves[count])][extractTarget(moveList->moves[count])] += depth;
            }
            
            // Keep this as the current best score and discard scores worse than this.
            alpha = score; // (defined as) PV Node

            // Write PV move
            pvTable[ply][ply] = moveList->moves[count];

            // Loop over all the deeper ply's
            for(int nextPly = ply+1 ; nextPly < pvLength[ply+1] ; nextPly++){
                // Copy move from deeper ply into the current ply's line
                pvTable[ply][nextPly] = pvTable[ply+1][nextPly];
            }

            // Adjust pvLength
            pvLength[ply] = pvLength[ply+1];
            
            // Fail-Hard Beta Cutoff
            // Kept inside score > alpha for performance increase
            if (score >= beta){
                // Store the move with eval = beta
                writeToHashTable(beta , depth , hashBETA);

                // Store Killer Moves -> Defined only for quite moves
                if(extractCaptureFlag(moveList->moves[count]) == 0){
                    killerMoves[1][ply] = killerMoves[0][ply];
                    killerMoves[0][ply] = moveList->moves[count];
                }

                return beta; // Fail High Case => Found Worse Move
            } 
          
        } 
    }
    
    if(legalMoves == 0){
        // King is in Check
        if(isCheck){
            // the +ply helps fine the checkmate , else in higher depth it goes astray
            // Assuming closest distance to the mating position
            return -MATE_VALUE + ply;
        }
        else{
            // Return Stalemate Score
            return 0;
        }

    }

    // Store the move withe eval = beta
    writeToHashTable(alpha , depth , hashFlag); // Move might be PV or Alpha , can't say for sure

    // Move Fails Low
    return alpha;

}


void searchPosition(int depth){

    // Init Score
    int score = 0;

    // Reset Node Counter
    searchNodes = 0;

    // Reset "time is up" flag
    stopped = 0;

    // Reset PV Flags
    followPV = 0;
    scorePV = 0;

    // Clear all the data Strutures for search for efficiency and redundancy
    memset(killerMoves , 0 , sizeof(killerMoves));
    memset(historyMoves , 0 , sizeof(historyMoves));
    memset(pvTable , 0 , sizeof(pvTable));
    memset(pvLength , 0 , sizeof(pvLength));
    

    // Implement Aspiration window initail aplha , beta
    int alpha = -INFINITY; 
    int beta = INFINITY;


    // Iterative Deepening
    for(int currDepth = 1; currDepth <= depth; currDepth++){

        if (stopped == 1) {
            // Stop calculating and return the best move found so far
            break;
        }

        // Enable Follow PV
        followPV = 1;

        // 50000 is the placeholder for infinity here
        score = negaMaxSearch(alpha , beta , currDepth);
        
        // Aspiration Window Method
        if ((score <= alpha) || (score >= beta)){
            // Fell out of window , try again with full window search
            // printf("Depth = %d, score = %d\n" , currDepth , score);
            alpha = -INFINITY; 
            beta = INFINITY;
            continue;
        }
        // Set up window for next search
        alpha = score - 50; 
        beta = score + 50;
        if(pvLength[0]){
            if ((score > -MATE_VALUE) && (score < -MATE_SCORE)){
                printf("info score mate %d depth %d nodes %llu time %d pv ", -(score + MATE_VALUE) / 2 - 1 , currDepth , searchNodes , getTimeInMilliSeconds() - startTime);
            }
            else if ((score > MATE_SCORE) && (score < MATE_VALUE)){
                printf("info score mate %d depth %d nodes %llu time %d pv ", (MATE_VALUE - score) / 2 + 1 , currDepth , searchNodes , getTimeInMilliSeconds() - startTime);
            }
            else{
                printf("info score cp %d depth %d nodes %llu time %d pv ", score , currDepth , searchNodes , getTimeInMilliSeconds() - startTime);
            }
        }

        for(int count = 0; count < pvLength[0]; count++){
            // Print PV Move
            printMove(pvTable[0][count]);
            printf(" ");
        }
        printf("\n");
    }

    printf("bestmove ");
    printMove(pvTable[0][0]);
    printf("\n");
}

/*
    UCI - UNIVERSAL CHESS INTERFACE (For GUI and communication to play to Gui)
    
    Position Commands : 
    
    1) position startpos : Init Start Position
    2) position startpos e4e5 e7e5 : Init Start Position and make the moves
    3) position fen someFENString : Init Posiiton given by the someFENString
    4) position fen someFENString e4e5 e7e5 : Init Posiiton given by the someFENString and make these moves

    Go Commands : 

    1) go depth 6
*/

// Parse User/GUI move string (eg : "e7e8q")
int parseMoveString(char * moveString){

    // Init Move List 
    Move moveList[1];

    generateMoves(moveList);

    // Parse source Square
    int srcSquare =  8 * (8 - (moveString[1] - '0')) + (moveString[0] - 'a');
    int targetSquare =  8 * (8 - (moveString[3] - '0')) + (moveString[2] - 'a');

    for(int count = 0; count < moveList->moveCount; count++){
        int move = moveList->moves[count];
        if((srcSquare == extractSource(move)) && (targetSquare == extractTarget(move))){
            // init promoted piece
            int promotedPiece = extractPromotedPiece(move);
            if (promotedPiece){
                if ((promotedPiece == Q || promotedPiece == q) && moveString[4] == 'q'){
                    // Legal Move
                    return move;
                }
                else if ((promotedPiece == R || promotedPiece == r) && moveString[4] == 'r'){
                    // Legal Move
                    return move;
                }
                else if ((promotedPiece == B || promotedPiece == b) && moveString[4] == 'b'){
                    // Legal Move
                    return move;
                }
                else if ((promotedPiece == N || promotedPiece == n) && moveString[4] == 'n'){
                    // Legal Move
                    return move;
                }

                // if promotion not in the option ('q' , 'r' , 'b' , 'n') in string
                continue; // So it can go to the Illegal part and return 0
            }
            // Legal Move
            return move;
        }
    }

    // Return Illegal Move
    return 0;
}

void parsePosition(char * command){
    
    // Skip the position key word and go to next token
    command += 9;

    char * currCmd = command;
    
    if(strncmp(command , "startpos" , 8) == 0){
        parseFENString(START_POSITION);
    }
    
    else{
        // Make ure FEN command present in the string
        currCmd = strstr(command , "fen");
        if(currCmd == NULL){
            parseFENString(START_POSITION);
        }
        else{
            // fen present
            currCmd += 4; // Go to the token after fen and space
            parseFENString(currCmd);
        }
    }
    
    // Parse the moves after getting a position
    currCmd = strstr(command , "moves");
    
    if(currCmd != NULL){
    
        currCmd += 6;
    
        while(*currCmd){
    
            // Parse Next Move
            int move = parseMoveString(currCmd);

            if (move == 0) break;

            // Increment Repetition Index
            repetitionIndex++;
            repetitionTable[repetitionIndex] = hashKey;

            makeMove(move  , allMoves);

            while(*currCmd && *currCmd != ' ') currCmd++;
            currCmd++;
    
        }
    }
    
    printBoard();
}

void resetTimeControl(){
    // Reset Timing
    quit = 0;
    movesToGo = 40;
    moveTime = -1;
    time = -1;
    inc = 0;
    startTime = 0;
    stopTime = 0;
    timeSet = 0;
    stopped = 0;
}

// parse UCI command "go"
void parseGo(char *command)
{
    resetTimeControl();

    // init parameters
    int depth = -1;

    // init argument
    char *argument = NULL;

    // infinite search
    if ((argument = strstr(command,"infinite"))) {}

    // match UCI "binc" command
    if ((argument = strstr(command,"binc")) && (sideToMove == BLACK))
        // parse black time increment
        inc = atoi(argument + 5);

    // match UCI "winc" command
    if ((argument = strstr(command,"winc")) && (sideToMove == WHITE))
        // parse white time increment
        inc = atoi(argument + 5);

    // match UCI "wtime" command
    if ((argument = strstr(command,"wtime")) && (sideToMove == WHITE))
        // parse white time limit
        time = atoi(argument + 6);

    // match UCI "btime" command
    if ((argument = strstr(command,"btime")) && (sideToMove == BLACK))
        // parse black time limit
        time = atoi(argument + 6);

    // match UCI "movestogo" command
    if ((argument = strstr(command,"movestogo")))
        // parse number of moves to go
        movesToGo = atoi(argument + 10);

    // match UCI "movetime" command
    if ((argument = strstr(command,"movetime")))
        // parse amount of time allowed to spend to make a move
        movesToGo = atoi(argument + 9);

    // match UCI "depth" command
    if ((argument = strstr(command,"depth")))
        // parse search depth
        depth = atoi(argument + 6);

    // if move time is not available
    if(moveTime != -1)
    {
        // set time equal to move time
        time = moveTime;

        // set moves to go to 1
        movesToGo = 1;
    }

    // init start time
    startTime = getTimeInMilliSeconds();

    // init search depth
    depth = depth;

    // if time control is available
    if(time != -1)
    {
        // flag we're playing with time control
        timeSet = 1;

        // set up timing
        time /= movesToGo;

        // Illegal (empty) move bug fix
        if (time > 1500) time -= 50;
        time -= 50;
        stopTime = startTime + time + inc;
    }

    // if depth is not available
    if(depth == -1)
        // set depth to 64 plies (takes ages to complete...)
        depth = 64;

    // print debug info
    printf("time:%d start:%u stop:%u depth:%d timeset:%d\n",
    time, startTime, stopTime, depth, timeSet);

    // search position
    searchPosition(depth);
}

void mainUCILoop(){

    // Max TT size
    int maxHash = 128;

    // default MB value
    int mb = 64;
    // reset STDIN and STDOUT buffers
    setbuf(stdin , NULL);
    setbuf(stdout , NULL);

    // define user/GUI input buffer 
    char input[2000];

    // Print Engine Info
    printf("ID Name : BBC %s\n" , VERSION);
    printf("ID Author : Ashvin Ganesh\n");
    printf("option name Hash type spin default 64 min 4 max %d\n" , maxHash);
    printf("uciok\n");

    // Main Loop
    while(1){
        // reset user/GUI input
        memset(input , 0 , sizeof(input));

        // Make sure output reaches the gui
        fflush(stdout);

        // get user/GUI Input

        if(!(fgets(input , 2000 , stdin))) continue;
        if(input[0] == '\n') continue;

        // Parse UCI "isready" command
        if (strncmp(input , "isready" , 7) == 0){
            printf("readyok\n");
            continue;
        }
        // Parse UCI "position" command
        else if (strncmp(input , "position" , 8) == 0){
            parsePosition(input);
            clearTranspositionTable();
        }
        // Parse UCI "ucinewgame" command
        else if (strncmp(input , "ucinewgame" , 10) == 0){
            parsePosition("position startpos");
            clearTranspositionTable();
        }
        // Parse UCI "go" command
        else if (strncmp(input , "go" , 2) == 0){
            parseGo(input);
        }
        // Parse UCI "quit" command
        else if (strncmp(input , "quit" , 4) == 0){
            break;
        }
        // Parse UCI "quit" command
        else if (strncmp(input , "uci" , 3) == 0){
            // Print Engine Info
            printf("ID Name : BBC\n");
            printf("ID Name : Ashvin Ganesh\n");
            printf("uciok\n");
        }
        else if(!(strncmp(input , "setoption name Hash value" , 26))){
            // Init MB
            sscanf(input,  "%*s %*s %*s %*s %d" , &mb);

            // Adjust MB if it is going beyond limits
            if(mb < 4) mb = 4;
            if(mb > maxHash) mb = maxHash;

            // Set Hash table Size in MB
            printf("Set hash table size to %dMB\n" , mb);
            initTranspositionTable(mb);
        }

    }
}

/*
    PERFT TESTING
    Leaf Node : No. of positions reached durint the test of the move generator at a given depth
*/

U64 nodes;

// PERFT DRIVER
static inline void perftDriver(int depth){
    // Recursive base condition
    if (depth == 0) {
        nodes++;
        return;
    }
    
    Move moveList[1];
    generateMoves(moveList);

    for(int count = 0; count < moveList->moveCount; count++){
        copyBoardState();

        // Make the Move
        if (!makeMove(moveList->moves[count] , allMoves)) continue; // Skip if Illegal

        perftDriver(depth - 1);
        // revert position
        restoreBoardState();

        // U64 newPostionHashKey = generateZobristHashKeys();

        // if (hashKey != newPostionHashKey){
        //     printf("Make Move : \n");
        //     printf("Move : ");
        //     printMove(moveList->moves[count]);
        //     printBoard();
        //     printf("Hash key should be : %llx\n" , newPostionHashKey);
        //     getchar();
        // }
    }
}

// PERFT TEST
static inline void perftTest(int depth){

    printf(" Performance Test \n");

    Move moveList[1];
    generateMoves(moveList);

    int start = getTimeInMilliSeconds();

    for(int count = 0; count < moveList->moveCount; count++){
        copyBoardState();

        // Make the Move
        if (!makeMove(moveList->moves[count] , allMoves)) continue; // Skip if Illegal
        long cummulativeNodes = nodes;

        perftDriver(depth - 1);
        // revert position
        long oldNodes = nodes - cummulativeNodes;
        restoreBoardState();
        printf(" Move : %s %s %c - Nodes : %ld , Piece Moved : %s \n", squareToCoordinates[extractSource(moveList->moves[count])],
                       squareToCoordinates[extractTarget(moveList->moves[count])],
                       promotedPieces[extractPromotedPiece(moveList->moves[count])],
                       oldNodes,
                       unicodePieces[extractMovedPiece(moveList->moves[count])]);
    }
    printf("   Depth = %d\n" , depth);
    printf("   Nodes = %lld\n" , nodes);
    printf("   Time =  %d\n" , getTimeInMilliSeconds() - start);
}

/*
    INITALIZE ALL FUNCTIONS
*/

void init(){
    generatePawnAttacks();
    generateKnightAttacks();
    generateKingAttacks();
    // initMagicNumbers();
    initBishopAndRookAttacks(bishop);
    initBishopAndRookAttacks(rook);
    initRandomKeys(); // For Hashing
    initTranspositionTable(12); // 12MB Hash Table
    initEvaluationMasks();
}

/*
    MAIN DRIVER
*/

int main(){
    init();
    int debug = 0;

    if(debug){
        // e2a6 b4c3
        parseFENString(START_POSITION); 
        printBoard();
        printf("Score : %d\n" , evaluate());
        searchPosition(10);
    }
    else{
        mainUCILoop();
    }
    return 0;
}
