




static inline int countBits(U64 bitBoard){
int count = 0;
while(bitBoard){
count++;
bitBoard &= (bitBoard - 1);
}

return count;

}

static inline int getLS1BIndex(U64 bitBoard){
if (bitBoard) return countBits((bitBoard & -bitBoard) - 1);
return -1;
}



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

enum{
WHITE , BLACK , BOTH
};

enum { rook, bishop };

enum{
WK = 1 , WQ = 2 , BK = 4 , BQ = 8
};

enum {
P , N , B , R , Q , K , 
p , n , b , r , q , k
};

enum {allMoves , captureMoves};



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



char asciiPieces[] = "PNBRQKpnbrqk";

char *unicodePieces[12] = {
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

U64 pieceBitBoards[12];

U64 boardOccupancies[3];

int sideToMove;

int enPassant = NULL_SQUARE;

int castlingRights;

U64 repetitionTable[1000]; int repetitionIndex;

int ply; 

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



U64 hashKey;

U64 pieceKeys[12][64];

U64 enPassantKeys[64];

U64 castlingKeys[16]; U64 sideToMoveKey;

void initRandomKeys(){

for(int piece = P; piece <= k; piece++){
for(int square = 0; square < 64; square++){

pieceKeys[piece][square] = getRandomU64Number();

}
}

for(int square = 0; square < 64; square++){

enPassantKeys[square] = getRandomU64Number();

}

for(int idx = 0; idx < 16; idx ++){

castlingKeys[idx] = getRandomU64Number();

}

sideToMoveKey = getRandomU64Number();

}

U64 generateZobristHashKeys(){

U64 positionKey = 0ULL;

U64 copyBitBoard;

for(int piece = P; piece <= k; piece++){

copyBitBoard = pieceBitBoards[piece];

while(copyBitBoard){
int square = getLS1BIndex(copyBitBoard);

positionKey ^= pieceKeys[piece][square];

popBitOnSquare(copyBitBoard , square);

}

}


if (enPassant != NULL_SQUARE){

positionKey ^= enPassantKeys[enPassant];
}

positionKey ^= castlingKeys[castlingRights];

if (sideToMove == BLACK) positionKey ^= sideToMoveKey;

return positionKey;
}



void printBitBoard(U64 bitBoard){
printf("\n");
for(int rank = 0; rank < 8 ;rank++){
for(int file = 0; file < 8; file++){
int square = rank * 8 + file;
if (!file){
printf("%d " , 8-rank);
}
printf(" %d" , getBitOnSquare(bitBoard , square) ? 1 : 0);
}
printf("\n");
}
printf("\n   a b c d e f g h \n");

printf("\n   Bitboard = %llud \n" , bitBoard);

}

void printBoard(){
printf("\n");

for(int rank = 0 ; rank < 8; rank++){
for(int file = 0; file < 8; file++){
int square = 8 * rank + file;

if(!file) printf("%d " , (8 - rank));

int piece = -1;

for(int consideredPiece = P; consideredPiece <=k ; consideredPiece++){
if(getBitOnSquare(pieceBitBoards[consideredPiece] , square)) piece = consideredPiece;
}
printf(" %c " , ((piece == -1) ? ' .' : asciiPieces[piece]));
printf(" %s" , ((piece == -1) ? "." : unicodePieces[piece]));
}
printf("\n");
}
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



const int bishopOccupancyBits[64] = {
6 , 5 , 5 , 5 , 5 , 5 , 5 , 6 , 
5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 
5 , 5 , 7 , 7 , 7 , 7 , 5 , 5 , 
5 , 5 , 7 , 9 , 9 , 7 , 5 , 5 , 
5 , 5 , 7 , 9 , 9 , 7 , 5 , 5 , 
5 , 5 , 7 , 7 , 7 , 7 , 5 , 5 , 
5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 
6 , 5 , 5 , 5 , 5 , 5 , 5 , 6 
};

const int rookOccupancyBits[64] = {
12 , 11 , 11 , 11 , 11 , 11 , 11 , 12 , 
11 , 10 , 10 , 10 , 10 , 10 , 10 , 11 , 
11 , 10 , 10 , 10 , 10 , 10 , 10 , 11 , 
11 , 10 , 10 , 10 , 10 , 10 , 10 , 11 , 
11 , 10 , 10 , 10 , 10 , 10 , 10 , 11 , 
11 , 10 , 10 , 10 , 10 , 10 , 10 , 11 , 
11 , 10 , 10 , 10 , 10 , 10 , 10 , 11 , 
12 , 11 , 11 , 11 , 11 , 11 , 11 , 12
};



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



void parseFENString(char* fen){

memset(pieceBitBoards , 0ULL , sizeof(pieceBitBoards));
memset(boardOccupancies , 0ULL , sizeof(boardOccupancies));
sideToMove = 0;
enPassant = NULL_SQUARE;
castlingRights = 0;

repetitionIndex = 0;

memset(repetitionTable , 0 , sizeof(repetitionTable));

for(int rank = 0; rank < 8; rank++){

for(int file = 0; file < 8; file++){
int square = 8 * rank + file;

if (((*fen >= 'a') && (*fen <= 'z')) || ((*fen >= 'A') && (*fen <= 'Z'))){
int piece = asciiToIntegerMapping[*fen];

setBitOnSquare(pieceBitBoards[piece], square);

fen++;
}

if ((*fen >= '0') && (*fen <='9')){
int offset = *fen - '0';

int piece = -1;

for(int consideredPiece = P; consideredPiece <=k ; consideredPiece++){
if(getBitOnSquare(pieceBitBoards[consideredPiece] , square)) piece = consideredPiece;
}

if (piece == -1) file--;

file += offset;

fen++;
}

if((*fen == '/')) fen++;
} 
}
fen++;

(*fen == 'w') ? (sideToMove = WHITE) : (sideToMove = BLACK) ;

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
fen++;
}

fen++;
if (*fen != '-'){
int file = fen[0] - 'a';
int rank = 8 - (fen[1] - '0'); 

enPassant = 8 * rank + file;
}
else{
enPassant = NULL_SQUARE;
}
for(int piece = P; piece <= K; piece++){
boardOccupancies[WHITE] |= pieceBitBoards[piece];
}

for(int piece = p; piece <= k; piece++){
boardOccupancies[BLACK] |= pieceBitBoards[piece];
}

boardOccupancies[BOTH] |= boardOccupancies[WHITE];
boardOccupancies[BOTH] |= boardOccupancies[BLACK];

hashKey = generateZobristHashKeys();
}




U64 pawnAttacks[2][64];

U64 knightAttacks[64];

U64 kingAttacks[64];

U64 bishopAttacksMask[64];

U64 rookAttacksMask[64];

U64 bishopAttacks[64][512];

U64 rookAttacks[64][4096];
U64 maskPawnAttacks(int square , int sideToMove){
U64 pawnBB = 0ULL;

U64 attacksBB = 0ULL;

setBitOnSquare(pawnBB , square);

if(!sideToMove){
if ((pawnBB >> 7) & maskFileA) attacksBB |= (pawnBB >> 7);
if ((pawnBB >> 9) & maskFileH) attacksBB |= (pawnBB >> 9);
}
else{
if ((pawnBB << 7) & maskFileH) attacksBB |= (pawnBB << 7);
if ((pawnBB << 9) & maskFileA) attacksBB |= (pawnBB << 9);
}
return attacksBB;
}

void generatePawnAttacks(){
for(int square = 0; square < 64; square++){
pawnAttacks[WHITE][square] = maskPawnAttacks(square , WHITE);
pawnAttacks[BLACK][square] = maskPawnAttacks(square , BLACK);
}
}


U64 maskKnightAttacks(int square){


U64 knightBB = 0ULL;

U64 attacksBB = 0ULL;

setBitOnSquare(knightBB , square);
if ((knightBB >> 10) & maskFileGH) attacksBB |= (knightBB >> 10);
if ((knightBB >> 6) & maskFileAB) attacksBB |= (knightBB >> 6);
if ((knightBB >> 17) & maskFileH) attacksBB |= (knightBB >> 17);
if ((knightBB >> 15) & maskFileA) attacksBB |= (knightBB >> 15);
if ((knightBB << 10) & maskFileAB) attacksBB |= (knightBB << 10);
if ((knightBB << 6) & maskFileGH) attacksBB |= (knightBB << 6);
if ((knightBB << 17) & maskFileA) attacksBB |= (knightBB << 17);
if ((knightBB << 15) & maskFileH) attacksBB |= (knightBB << 15);

return attacksBB;
}

void generateKnightAttacks(){
for (int square = 0; square < 64; square++){
knightAttacks[square] = maskKnightAttacks(square);
}
}

U64 maskKingAttacks(int square){

U64 kingBB = 0ULL;

U64 attacksBB = 0ULL;
setBitOnSquare(kingBB , square);
if (kingBB >> 8) attacksBB |= (kingBB >> 8); 
if ((kingBB >> 9) & maskFileH) attacksBB |= (kingBB >> 9); 
if ((kingBB >> 7) & maskFileA) attacksBB |= (kingBB >> 7); 
if ((kingBB >> 1) & maskFileH) attacksBB |= (kingBB >> 1); 
if (kingBB << 8) attacksBB |= (kingBB << 8); 
if ((kingBB << 9) & maskFileA) attacksBB |= (kingBB << 9); 
if ((kingBB << 7) & maskFileH) attacksBB |= (kingBB << 7); 
if ((kingBB << 1) & maskFileA) attacksBB |= (kingBB << 1); 
return attacksBB;
}

void generateKingAttacks(){
for (int square = 0; square < 64; square++){
kingAttacks[square] = maskKingAttacks(square);
}
}

U64 maskBishopAttacks(int square){

U64 attacksBB = 0ULL;

int currRank = square / 8;
int currFile = square % 8;
int r , f;
for(r = currRank + 1 , f = currFile + 1 ; r <= 6 && f <= 6; r++, f++) attacksBB |= (1ULL << (8 * r + f ));
for(r = currRank - 1 , f = currFile + 1 ; r >= 1 && f <= 6; r--, f++) attacksBB |= (1ULL << (8 * r + f ));
for(r = currRank + 1 , f = currFile - 1 ; r <= 6 && f >= 1; r++, f--) attacksBB |= (1ULL << (8 * r + f ));
for(r = currRank - 1 , f = currFile - 1 ; r >= 1 && f >= 1; r--, f--) attacksBB |= (1ULL << (8 * r + f ));
return attacksBB;
}

U64 generateBishopAttacksWithObstacles(int square , U64 boardState){
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
return attacksBB;
}

U64 maskRookAttacks(int square){
U64 attacksBB = 0ULL;

int currRank = square / 8;
int currFile = square % 8;
int r , f;
for(r = currRank + 1 ; r <= 6 ; r++) attacksBB |= (1ULL << (8 * r + currFile )); for(r = currRank - 1 ; r >= 1 ; r--) attacksBB |= (1ULL << (8 * r + currFile )); for(f = currFile + 1 ; f <= 6 ; f++) attacksBB |= (1ULL << (8 * currRank + f )); for(f = currFile - 1 ; f >= 1 ; f--) attacksBB |= (1ULL << (8 * currRank + f )); return attacksBB;
}

U64 generateRookAttacksWithObstacles(int square , U64 boardState){

U64 attacksBB = 0ULL;

int currRank = square / 8;
int currFile = square % 8;
int r , f;
for(r = currRank + 1 ; r <= 7 ; r++)
{
attacksBB |= (1ULL << (8 * r + currFile )); if((1ULL << (8 * r + currFile)) & boardState) break;
} 
for(r = currRank - 1 ; r >= 0 ; r--) 
{
attacksBB |= (1ULL << (8 * r + currFile )); if((1ULL << (8 * r + currFile )) & boardState) break;
} 
for(f = currFile + 1 ; f <= 7 ; f++) 
{
attacksBB |= (1ULL << (8 * currRank + f )); if((1ULL << (8 * currRank + f )) & boardState) break;
} 
for(f = currFile - 1 ; f >= 0 ; f--) 
{
attacksBB |= (1ULL << (8 * currRank + f )); if((1ULL << (8 * currRank + f )) & boardState) break;
} 
return attacksBB;
}

U64 setOccupancy(int index , int bitCount , U64 attackMask){
U64 occupancy = 0ULL;

for(int count = 0; count < bitCount; count++){

int square = getLS1BIndex(attackMask);

popBitOnSquare(attackMask , square);

if ((index & (1 << count))){
occupancy |= (1ULL << square);
}
}
return occupancy;
}

void initBishopAndRookAttacks(int isBishopOrRook){
for(int square = 0; square < 64; square++){
bishopAttacksMask[square] = maskBishopAttacks(square);
rookAttacksMask[square] = maskRookAttacks(square);

U64 attackMask = isBishopOrRook
? bishopAttacksMask[square]
: rookAttacksMask[square];

int relevantBits = countBits(attackMask);

int occupancyIndex = (1 << relevantBits);

for(int index = 0; index < occupancyIndex; index++){

if(isBishopOrRook){
U64 occupancy = setOccupancy(index , relevantBits , attackMask);

int magicIndex = (occupancy * bishopMagicNumbers[square]) >> (64 - bishopOccupancyBits[square]);

bishopAttacks[square][magicIndex] = generateBishopAttacksWithObstacles(square , occupancy);

}
else{
U64 occupancy = setOccupancy(index , relevantBits , attackMask);

int magicIndex = (occupancy * rookMagicNumbers[square]) >> (64 - rookOccupancyBits[square]);

rookAttacks[square][magicIndex] = generateRookAttacksWithObstacles(square , occupancy);
}
}

}
}

static inline U64 getBishopAttacks(int square , U64 occupancy){
occupancy &= bishopAttacksMask[square];
occupancy *= bishopMagicNumbers[square];
occupancy >>= (64 - bishopOccupancyBits[square]);

return bishopAttacks[square][occupancy];

}

static inline U64 getRookAttacks(int square , U64 occupancy){
occupancy &= rookAttacksMask[square];
occupancy *= rookMagicNumbers[square];
occupancy >>= (64 - rookOccupancyBits[square]);

return rookAttacks[square][occupancy];
}

static inline U64 getQueenAttacks(int square , U64 occupancy){

return (getBishopAttacks(square, occupancy) | getRookAttacks(square, occupancy));;
}

U64 generateMagicNumberCandidates(){
return getRandomU64Number() & getRandomU64Number() & getRandomU64Number();
}



U64 getMagicNumber(int square , int relevantBits , int isBishopOrRook){


U64 occupancyMap[4096];

U64 attacks[4096];

U64 usedAttacks[4096];

U64 attackMask = isBishopOrRook 
? maskBishopAttacks(square) 
: maskRookAttacks(square);
int occupancyIndex = 1 << relevantBits;

for (int index = 0; index < occupancyIndex; index++){
occupancyMap[index] = setOccupancy(index , relevantBits , attackMask);

attacks[index] = isBishopOrRook 
? generateBishopAttacksWithObstacles(square , occupancyMap[index])
: generateRookAttacksWithObstacles(square , occupancyMap[index]); 
}

for (int randomCount = 0; randomCount < 100000000; randomCount++){
U64 magicNumberCandidate = generateMagicNumberCandidates();

if (countBits((attackMask * magicNumberCandidate) & 0xFF00000000000000ULL) < 6) continue;

memset(usedAttacks , 0ULL , sizeof(usedAttacks)); int index, failFlag;

for(index = 0 , failFlag = 0; !failFlag && index < occupancyIndex; index++){
int magicIndex = (int)((occupancyMap[index] * magicNumberCandidate) >> (64 - relevantBits));

if (usedAttacks[magicIndex] == 0ULL){
usedAttacks[magicIndex] = attacks[index];
}
else if (usedAttacks[magicIndex] != attacks[index]){
failFlag = 1;
}
}
if (!failFlag){
return magicNumberCandidate;
}
}

printf(" Magic Number Not Found !!! \n");
return 0ULL;

}

void initMagicNumbers(){
for(int square = 0; square < 64; square++){
rookMagicNumbers[square] = getMagicNumber(square , rookOccupancyBits[square] , rook);
printf(" 0x%llxULL ,\n" , rookMagicNumbers[square]);
}
for(int square = 0; square < 64; square++){
bishopMagicNumbers[square] = getMagicNumber(square , bishopOccupancyBits[square] , bishop);
}
}



(source) | \
(target << 6) | \
(piece << 12) | \
(promoted << 16) | \
(capture << 20) | \
(double << 21) | \
(enpassant << 22) | \
(castling << 23) \

typedef struct {
int moves[256]; int moveCount; }Move;

void printMove(int move){
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
moveList->moves[moveList->moveCount] = move; 
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
printf("\t%s-%s%c   %s        %d        %d          %d          %d\n", squareToCoordinates[extractSource(move)],
squareToCoordinates[extractTarget(move)],
extractPromotedPiece(move) ? promotedPieces[extractPromotedPiece(move)] : ' ',
asciiPieces[extractMovedPiece(move)],
extractCaptureFlag(move) ? 1 : 0,
extractDoublePushFlag(move) ? 1 : 0,
extractEnPassantFlag(move) ? 1 : 0,
extractCastlingFlag(move) ? 1 : 0);
printf("\t%s-%s%c   %s        %d        %d          %d          %d\n", squareToCoordinates[extractSource(move)],
squareToCoordinates[extractTarget(move)],
extractPromotedPiece(move) ? promotedPieces[extractPromotedPiece(move)] : ' ',
unicodePieces[extractMovedPiece(move)],
extractCaptureFlag(move) ? 1 : 0,
extractDoublePushFlag(move) ? 1 : 0,
extractEnPassantFlag(move) ? 1 : 0,
extractCastlingFlag(move) ? 1 : 0);
}
printf("\n Total Number Of Moves : %d\n" , moveList->moveCount);
}



U64 pieceBitBoardsCopy[12] , boardOccupanciesCopy[3]; \
int sideToMoveCopy , enPassantCopy , castlingRightsCopy; \
memcpy(pieceBitBoardsCopy , pieceBitBoards , 96); \
memcpy(boardOccupanciesCopy , boardOccupancies , 24); \
sideToMoveCopy = sideToMove , enPassantCopy = enPassant , castlingRightsCopy = castlingRights; \
U64 hashKeyCopy = hashKey; \

memcpy(pieceBitBoards , pieceBitBoardsCopy , 96); \
memcpy(boardOccupancies , boardOccupanciesCopy , 24); \
sideToMove = sideToMoveCopy , enPassant = enPassantCopy , castlingRights = castlingRightsCopy; \
hashKey = hashKeyCopy; \



static inline int isSquareAttacked(int square , int side){

if((side == WHITE) && (pawnAttacks[BLACK][square] & pieceBitBoards[P])) return 1;

if((side == BLACK) && (pawnAttacks[WHITE][square] & pieceBitBoards[p])) return 1;

if (knightAttacks[square] & ((side == WHITE) ? pieceBitBoards[N] : pieceBitBoards[n])) return 1;

if (kingAttacks[square] & ((side == WHITE) ? pieceBitBoards[K] : pieceBitBoards[k])) return 1;

if (getBishopAttacks(square , boardOccupancies[BOTH]) & ((side == WHITE) ? pieceBitBoards[B] : pieceBitBoards[b])) return 1;

if (getRookAttacks(square , boardOccupancies[BOTH]) & ((side == WHITE) ? pieceBitBoards[R] : pieceBitBoards[r])) return 1;

if (getQueenAttacks(square , boardOccupancies[BOTH]) & ((side == WHITE) ? pieceBitBoards[Q] : pieceBitBoards[q])) return 1;

return 0;
}

void printAllAttackedSquares(int side){
printf("\n");
for(int rank = 0; rank < 8; rank++){
for(int file = 0; file < 8; file++){
int square = 8 * rank + file;

if(!file) printf(" %d " , (8-rank));
printf(" %d" , isSquareAttacked(square , side) ? 1 : 0);
}
printf("\n");
}
printf("\n    a b c d e f g h \n");
}

static inline void generateMoves(Move * moveList){


moveList->moveCount = 0;

int srcSquare , targetSquare;

U64 bitBoardCopyOfPiece , attacks;

for(int piece = P; piece <= k; piece++){
bitBoardCopyOfPiece = pieceBitBoards[piece];

if (sideToMove == WHITE){
if (piece == P){
while(bitBoardCopyOfPiece){
srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

targetSquare = srcSquare - 8; 

if(!(targetSquare < a8) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare))){
if((srcSquare >= a7) && (srcSquare <= h7)){

addMove(moveList , encodeMove(srcSquare , targetSquare , piece , Q , 0 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , R , 0 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , B , 0 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , N , 0 , 0 , 0 , 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0 , 0));

if(((srcSquare >= a2) && (srcSquare <= h2)) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare - 8))){

addMove(moveList , encodeMove(srcSquare , targetSquare - 8 , piece , 0 , 0 , 1 , 0 , 0));
} 
}

}

attacks = pawnAttacks[sideToMove][srcSquare] & boardOccupancies[BLACK];

while (attacks){
targetSquare = getLS1BIndex(attacks);
if((srcSquare >= a7) && (srcSquare <= h7)){

addMove(moveList , encodeMove(srcSquare , targetSquare , piece , Q , 1 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , R , 1 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , B , 1 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , N , 1 , 0 , 0 , 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0 , 0));

}

popBitOnSquare(attacks , targetSquare);
}

if (enPassant != NULL_SQUARE){
U64 enpassantAttacks = pawnAttacks[sideToMove][srcSquare] & (1ULL << enPassant);
if (enpassantAttacks){
int enPassantTargetSquare = getLS1BIndex(enpassantAttacks);

addMove(moveList , encodeMove(srcSquare , enPassantTargetSquare , piece , 0 , 1 , 0 , 1 , 0));
}
}
popBitOnSquare(bitBoardCopyOfPiece , srcSquare);

} 
}

if(piece == K){
if(castlingRights & WK){
if(!(getBitOnSquare(boardOccupancies[BOTH] , f1)) && !(getBitOnSquare(boardOccupancies[BOTH] , g1))){
if(!isSquareAttacked(e1 , BLACK) & !isSquareAttacked(f1 , BLACK)){

addMove(moveList , encodeMove(e1 , g1 , piece , 0 , 0 , 0 , 0, 1));
}
}
}

if(castlingRights & WQ){
if(!(getBitOnSquare(boardOccupancies[BOTH] , b1)) && !(getBitOnSquare(boardOccupancies[BOTH] , c1)) && !(getBitOnSquare(boardOccupancies[BOTH] , d1))){
if(!isSquareAttacked(e1 , BLACK) & !isSquareAttacked(d1 , BLACK)){

addMove(moveList , encodeMove(e1 , c1 , piece , 0 , 0 , 0 , 0, 1));
}
}
}
}

}
else{
if (piece == p){
while(bitBoardCopyOfPiece){
srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

targetSquare = srcSquare + 8; 

if(!(targetSquare > h1) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare))){
if((srcSquare >= a2) && (srcSquare <= h2)){

addMove(moveList , encodeMove(srcSquare , targetSquare , piece , q , 0 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , r , 0 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , b , 0 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , n , 0 , 0 , 0 , 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0 , 0));

if(((srcSquare >= a7) && (srcSquare <= h7)) && !(getBitOnSquare(boardOccupancies[BOTH] , targetSquare + 8))){
addMove(moveList , encodeMove(srcSquare , targetSquare + 8 , piece , 0 , 0 , 1 , 0 , 0));
} 
}

}

attacks = pawnAttacks[sideToMove][srcSquare] & boardOccupancies[WHITE];

while (attacks){
targetSquare = getLS1BIndex(attacks);
if((srcSquare >= a2) && (srcSquare <= h2)){

addMove(moveList , encodeMove(srcSquare , targetSquare , piece , q , 1 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , r , 1 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , b , 1 , 0 , 0 , 0));
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , n , 1 , 0 , 0 , 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0 , 0));

}

popBitOnSquare(attacks , targetSquare);
}

if (enPassant != NULL_SQUARE){
U64 enpassantAttacks = pawnAttacks[sideToMove][srcSquare] & (1ULL << enPassant);
if (enpassantAttacks){
int enPassantTargetSquare = getLS1BIndex(enpassantAttacks);
addMove(moveList , encodeMove(srcSquare , enPassantTargetSquare , piece , 0 , 1 , 0 , 1 , 0));
}
}

popBitOnSquare(bitBoardCopyOfPiece , srcSquare);

} 
}
if(piece == k){
if(castlingRights & BK){
if(!(getBitOnSquare(boardOccupancies[BOTH] , f8)) && !(getBitOnSquare(boardOccupancies[BOTH] , g8))){
if(!isSquareAttacked(e8 , WHITE) & !isSquareAttacked(f8 , WHITE)){
addMove(moveList , encodeMove(e8 , g8 , piece , 0 , 0 , 0 , 0, 1));
}
}
}

if(castlingRights & BQ){
if(!(getBitOnSquare(boardOccupancies[BOTH] , b8)) && !(getBitOnSquare(boardOccupancies[BOTH] , c8)) && !(getBitOnSquare(boardOccupancies[BOTH] , d8))){
if(!isSquareAttacked(e8 , WHITE) & !isSquareAttacked(d8 , WHITE)){
addMove(moveList , encodeMove(e8 , c8 , piece , 0 , 0 , 0 , 0, 1));
}
}
}
}

}

if((sideToMove == WHITE) ? (piece == N) : (piece == n)){
while (bitBoardCopyOfPiece)
{
srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

attacks = knightAttacks[srcSquare] & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

while (attacks){
targetSquare = getLS1BIndex(attacks);

if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
}

popBitOnSquare(attacks , targetSquare);
}
popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
}

}

if((sideToMove == WHITE) ? (piece == B) : (piece == b)){
while (bitBoardCopyOfPiece)
{
srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

attacks = getBishopAttacks(srcSquare , boardOccupancies[BOTH]) & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

while (attacks){
targetSquare = getLS1BIndex(attacks);

if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
}

popBitOnSquare(attacks , targetSquare);
}
popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
}

}

if((sideToMove == WHITE) ? (piece == R) : (piece == r)){
while (bitBoardCopyOfPiece)
{
srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

attacks = getRookAttacks(srcSquare , boardOccupancies[BOTH]) & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

while (attacks){
targetSquare = getLS1BIndex(attacks);

if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
}

popBitOnSquare(attacks , targetSquare);
}
popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
}

}

if((sideToMove == WHITE) ? (piece == Q) : (piece == q)){
while (bitBoardCopyOfPiece)
{
srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

attacks = getQueenAttacks(srcSquare , boardOccupancies[BOTH]) & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

while (attacks){
targetSquare = getLS1BIndex(attacks);

if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
}

popBitOnSquare(attacks , targetSquare);
}
popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
}

}

if((sideToMove == WHITE) ? (piece == K) : (piece == k)){
while (bitBoardCopyOfPiece)
{
srcSquare = getLS1BIndex(bitBoardCopyOfPiece);

attacks = kingAttacks[srcSquare] & ((sideToMove == WHITE) ? ~boardOccupancies[WHITE] : ~boardOccupancies[BLACK]);

while (attacks){
targetSquare = getLS1BIndex(attacks);

if(!getBitOnSquare(((sideToMove == WHITE) ? boardOccupancies[BLACK] : boardOccupancies[WHITE]) , targetSquare)){
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 0 , 0 , 0, 0));
}
else{
addMove(moveList , encodeMove(srcSquare , targetSquare , piece , 0 , 1 , 0 , 0, 0));
}

popBitOnSquare(attacks , targetSquare);
}
popBitOnSquare(bitBoardCopyOfPiece , srcSquare);
}

}
}
}



const int castlingRightsMask[64] = {
7 , 15 , 15 , 15 , 3 , 15 , 15 , 11 ,
15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
15 , 15 , 15 , 15 , 15 , 15 , 15 , 15 ,
13 , 15 , 15 , 15 , 12 , 15 , 15 , 14 
};

static inline int makeMove(int move , int moveFlag){
if (moveFlag == allMoves){
copyBoardState();

int srcSquare = extractSource(move);
int targetSquare = extractTarget(move);
int pieceMoved = extractMovedPiece(move);
int promotedPiece = extractPromotedPiece(move);
int captureFlag = extractCaptureFlag(move);
int doublePushFlag = extractDoublePushFlag(move);
int enPassantFlag = extractEnPassantFlag(move);
int castlingFlag = extractCastlingFlag(move);

popBitOnSquare(pieceBitBoards[pieceMoved] , srcSquare);
setBitOnSquare(pieceBitBoards[pieceMoved] , targetSquare);

hashKey ^= pieceKeys[pieceMoved][srcSquare]; hashKey ^= pieceKeys[pieceMoved][targetSquare]; if(captureFlag){
int startPiece , endPiece; 

if(sideToMove == WHITE){
startPiece = p;
endPiece = k;
}
else{
startPiece = P;
endPiece = K;
}

for(int consideredPiece = startPiece ; consideredPiece <= endPiece ; consideredPiece++){
if(getBitOnSquare(pieceBitBoards[consideredPiece] , targetSquare)){
popBitOnSquare(pieceBitBoards[consideredPiece] , targetSquare);

hashKey ^= pieceKeys[consideredPiece][targetSquare];
break; } 
}
}

if(promotedPiece){
if(sideToMove == WHITE){
popBitOnSquare(pieceBitBoards[P] , targetSquare);

hashKey ^= pieceKeys[P][targetSquare];
}
else{
popBitOnSquare(pieceBitBoards[p] , targetSquare);

hashKey ^= pieceKeys[p][targetSquare];
}

setBitOnSquare(pieceBitBoards[promotedPiece] , targetSquare);

hashKey ^= pieceKeys[promotedPiece][targetSquare];
}

if(enPassantFlag){
if (sideToMove == WHITE){
popBitOnSquare(pieceBitBoards[p] , targetSquare + 8);

hashKey ^= pieceKeys[p][targetSquare + 8];
}
else{
popBitOnSquare(pieceBitBoards[P] , targetSquare - 8);

hashKey ^= pieceKeys[P][targetSquare - 8];
}
}

if (enPassant != NULL_SQUARE) hashKey ^= enPassantKeys[enPassant];

enPassant = NULL_SQUARE;

if(doublePushFlag){
if (sideToMove == WHITE){
enPassant = targetSquare + 8;

hashKey ^= enPassantKeys[targetSquare + 8];
}
else{
enPassant = targetSquare - 8;
hashKey ^= enPassantKeys[targetSquare - 8];
}
}

if(castlingFlag){
switch (targetSquare)
{ 
case (g1):
popBitOnSquare(pieceBitBoards[R] , h1);
setBitOnSquare(pieceBitBoards[R] , f1);
hashKey ^= pieceKeys[R][h1]; hashKey ^= pieceKeys[R][f1]; break;
case (c1):
popBitOnSquare(pieceBitBoards[R] , a1);
setBitOnSquare(pieceBitBoards[R] , d1);
hashKey ^= pieceKeys[R][a1]; hashKey ^= pieceKeys[R][d1]; break;
case (g8):
popBitOnSquare(pieceBitBoards[r] , h8);
setBitOnSquare(pieceBitBoards[r] , f8);
hashKey ^= pieceKeys[r][h8]; hashKey ^= pieceKeys[r][f8]; break;
case (c8):
popBitOnSquare(pieceBitBoards[r] , a8);
setBitOnSquare(pieceBitBoards[r] , d8);
hashKey ^= pieceKeys[r][a8]; hashKey ^= pieceKeys[r][d8]; break; 
default:
break;
}
}

hashKey ^= castlingKeys[castlingRights];

castlingRights &= castlingRightsMask[srcSquare];
castlingRights &= castlingRightsMask[targetSquare];
hashKey ^= castlingKeys[castlingRights]; memset(boardOccupancies , 0ULL , 24);

for(int piece = P; piece <= K; piece++){
boardOccupancies[WHITE] |= pieceBitBoards[piece];
}

for(int piece = p; piece <= k; piece++){
boardOccupancies[BLACK] |= pieceBitBoards[piece];
}

boardOccupancies[BOTH] |= boardOccupancies[WHITE];
boardOccupancies[BOTH] |= boardOccupancies[BLACK];

sideToMove ^= 1; hashKey ^= sideToMoveKey;

if(isSquareAttacked((sideToMove == WHITE) ? getLS1BIndex(pieceBitBoards[k]) : getLS1BIndex(pieceBitBoards[K]), sideToMove)){

restoreBoardState();

return 0;
}
else{
return 1;
}

}
else{
if (extractCaptureFlag(move)){
makeMove(move , allMoves);
}
else{
return 0;
}
}
}





const int positionalPawnScore[64] = 
{
90, 90, 90, 90, 90, 90, 90, 90,
30, 30, 30, 40, 40, 30, 30, 30,
20, 20, 20, 30, 30, 30, 20, 20,
10, 10, 10, 20, 20, 10, 10, 10,
5, 5, 10, 20, 20, 5, 5, 5,
0, 0, 0, 5, 5, 0, 0, 0,
0, 0, 0, -10, -10, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0
};

const int positionalKnightScore[64] = 
{
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 10, 10, 0, 0, -5,
-5, 5, 20, 20, 20, 20, 5, -5,
-5, 10, 20, 30, 30, 20, 10, -5,
-5, 10, 20, 30, 30, 20, 10, -5,
-5, 5, 20, 10, 10, 20, 5, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, -10, 0, 0, 0, 0, -10, -5
};

const int positionalBishopScore[64] = 
{
0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 10, 10, 0, 0, 0,
0, 0, 10, 20, 20, 10, 0, 0,
0, 0, 10, 20, 20, 10, 0, 0,
0, 10, 0, 0, 0, 0, 10, 0,
0, 30, 0, 0, 0, 0, 30, 0,
0, 0, -10, 0, 0, -10, 0, 0

};

const int positionalRookScore[64] =
{
50, 50, 50, 50, 50, 50, 50, 50,
50, 50, 50, 50, 50, 50, 50, 50,
0, 0, 10, 20, 20, 10, 0, 0,
0, 0, 10, 20, 20, 10, 0, 0,
0, 0, 10, 20, 20, 10, 0, 0,
0, 0, 10, 20, 20, 10, 0, 0,
0, 0, 10, 20, 20, 10, 0, 0,
0, 0, 0, 20, 20, 0, 0, 0

};

const int positionalKingScore[64] = 
{
0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 5, 5, 5, 5, 0, 0,
0, 5, 5, 10, 10, 5, 5, 0,
0, 5, 10, 20, 20, 10, 5, 0,
0, 5, 10, 20, 20, 10, 5, 0,
0, 0, 5, 10, 10, 5, 0, 0,
0, 5, 5, -5, -5, 0, 5, 0,
0, 0, 5, 0, -15, 0, 10, 0
};

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



U64 fileMask[64];

U64 rankMask[64];

U64 isolatedPawnMask[64];

U64 whitePassedPawnMask[64];

U64 blackPassedPawnMask[64];

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

const int doublePawnPenalty = -10;

const int isolatedPawnPenalty = -10;

const int passedPawnAdvantage[8] = { 0, 10, 30, 50, 75, 100, 150, 200 };

const int semiOpenFileAdvantage = 10;

const int openFileAdvantage = 15;

const int kingShieldBonus = 5;

U64 setRankAndFileMask(int file , int rank){
U64 mask = 0ULL;

for(int r = 0; r < 8; r++){
for(int f = 0; f < 8; f++){
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

void initEvaluationMasks(){

for(int rank = 0; rank < 8; rank++){
for(int file = 0; file < 8; file++){
int square = 8 * rank + file;

fileMask[square] |= setRankAndFileMask(file , -1);

rankMask[square] |= setRankAndFileMask(-1 , rank);
}
}

for(int rank = 0; rank < 8; rank++){
for(int file = 0; file < 8; file++){
int square = 8 * rank + file;

isolatedPawnMask[square] |= setRankAndFileMask(file - 1 , -1);

isolatedPawnMask[square] |= setRankAndFileMask(file + 1 , -1);

}
}

for(int rank = 0; rank < 8; rank++){
for(int file = 0; file < 8; file++){
int square = 8 * rank + file;

whitePassedPawnMask[square] |= setRankAndFileMask(file - 1 , -1);

whitePassedPawnMask[square] |= setRankAndFileMask(file , -1);

whitePassedPawnMask[square] |= setRankAndFileMask(file + 1 , -1);

for(int r = 0; r < (8-rank); r++){
whitePassedPawnMask[square] &= ~(rankMask[(8 *(7-r) + file)]);
}

}
}

for(int rank = 0; rank < 8; rank++){
for(int file = 0; file < 8; file++){
int square = 8 * rank + file;

blackPassedPawnMask[square] |= setRankAndFileMask(file - 1 , -1);

blackPassedPawnMask[square] |= setRankAndFileMask(file , -1);

blackPassedPawnMask[square] |= setRankAndFileMask(file + 1 , -1);

for(int r = 0; r < (rank + 1); r++){
blackPassedPawnMask[square] &= ~(rankMask[(8 * r + file)]);
}

}
}
}



const int materialScore[2][12] =
{
82, 337, 365, 477, 1025, 12000, -82, -337, -365, -477, -1025, -12000,

94, 281, 297, 512, 936, 12000, -94, -281, -297, -512, -936, -12000
};

const int openingPhaseScore = 6192;
const int endgamePhaseScore = 518;

enum { OPENING, ENDGAME, MIDDLEGAME };

enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

const int positionalScore[2][6][64] =
{
0, 0, 0, 0, 0, 0, 0, 0,
98, 134, 61, 95, 68, 126, 34, -11,
-6, 7, 26, 31, 65, 56, 25, -20,
-14, 13, 6, 21, 23, 12, 17, -23,
-27, -2, -5, 12, 17, 6, 10, -25,
-26, -4, -4, -10, 3, 3, 33, -12,
-35, -1, -20, -23, -15, 24, 38, -22,
0, 0, 0, 0, 0, 0, 0, 0,

-167, -89, -34, -49, 61, -97, -15, -107,
-73, -41, 72, 36, 23, 62, 7, -17,
-47, 60, 37, 65, 84, 129, 73, 44,
-9, 17, 19, 53, 37, 69, 18, 22,
-13, 4, 16, 13, 28, 19, 21, -8,
-23, -9, 12, 10, 19, 17, 25, -16,
-29, -53, -12, -3, -1, 18, -14, -19,
-105, -21, -58, -33, -17, -28, -19, -23,

-29, 4, -82, -37, -25, -42, 7, -8,
-26, 16, -18, -13, 30, 59, 18, -47,
-16, 37, 43, 40, 35, 50, 37, -2,
-4, 5, 19, 50, 37, 37, 7, -2,
-6, 13, 13, 26, 34, 12, 10, 4,
0, 15, 15, 15, 14, 27, 18, 10,
4, 15, 16, 0, 7, 21, 33, 1,
-33, -3, -14, -21, -13, -12, -39, -21,

32, 42, 32, 51, 63, 9, 31, 43,
27, 32, 58, 62, 80, 67, 26, 44,
-5, 19, 26, 36, 17, 45, 61, 16,
-24, -11, 7, 26, 24, 35, -8, -20,
-36, -26, -12, -1, 9, -7, 6, -23,
-45, -25, -16, -17, 3, 0, -5, -33,
-44, -16, -20, -9, -1, 11, -6, -71,
-19, -13, 1, 17, 16, 7, -37, -26,

-28, 0, 29, 12, 59, 44, 43, 45,
-24, -39, -5, 1, -16, 57, 28, 54,
-13, -17, 7, 8, 29, 56, 47, 57,
-27, -27, -16, -16, -1, 17, -2, 1,
-9, -26, -9, -10, -2, -4, 3, -3,
-14, 2, -11, -2, -5, 2, 14, 5,
-35, -8, 11, 2, 8, 15, -3, 1,
-1, -18, -9, 10, -15, -25, -31, -50,

-65, 23, 16, -15, -56, -34, 2, 13,
29, -1, -20, -7, -8, -4, -38, -29,
-9, 24, 2, -16, -20, 6, 22, -22,
-17, -20, -12, -27, -30, -25, -14, -36,
-49, -1, -27, -39, -46, -44, -33, -51,
-14, -14, -22, -46, -44, -30, -15, -27,
1, 7, -8, -64, -43, -16, 9, 8,
-15, 36, 12, -54, 8, -28, 24, 14,


0, 0, 0, 0, 0, 0, 0, 0,
178, 173, 158, 134, 147, 132, 165, 187,
94, 100, 85, 67, 56, 53, 82, 84,
32, 24, 13, 5, -2, 4, 17, 17,
13, 9, -3, -7, -7, -8, 3, -1,
4, 7, -6, 1, 0, -5, -1, -8,
13, 8, 8, 10, 13, 0, 2, -7,
0, 0, 0, 0, 0, 0, 0, 0,

-58, -38, -13, -28, -31, -27, -63, -99,
-25, -8, -25, -2, -9, -25, -24, -52,
-24, -20, 10, 9, -1, -9, -19, -41,
-17, 3, 22, 22, 22, 11, 8, -18,
-18, -6, 16, 25, 16, 17, 4, -18,
-23, -3, -1, 15, 10, -3, -20, -22,
-42, -20, -10, -5, -2, -20, -23, -44,
-29, -51, -23, -15, -22, -18, -50, -64,

-14, -21, -11, -8, -7, -9, -17, -24,
-8, -4, 7, -12, -3, -13, -4, -14,
2, -8, 0, -1, -2, 6, 0, 4,
-3, 9, 12, 9, 14, 10, 3, 2,
-6, 3, 13, 19, 7, 10, -3, -9,
-12, -3, 8, 10, 13, 3, -7, -15,
-14, -18, -7, -1, 4, -9, -15, -27,
-23, -9, -23, -5, -9, -16, -5, -17,

13, 10, 18, 15, 12, 12, 8, 5,
11, 13, 13, 11, -3, 3, 8, 3,
7, 7, 7, 5, 4, -3, -5, -3,
4, 3, 13, 1, 2, 1, -1, 2,
3, 5, 8, 4, -5, -6, -8, -11,
-4, 0, -5, -1, -7, -12, -8, -16,
-6, -6, 0, 2, -9, -9, -11, -3,
-9, 2, 3, -1, -5, -13, 4, -20,

-9, 22, 22, 27, 27, 19, 10, 20,
-17, 20, 32, 41, 58, 25, 30, 0,
-20, 6, 9, 49, 47, 35, 19, 9,
3, 22, 24, 45, 57, 40, 57, 36,
-18, 28, 19, 47, 31, 34, 39, 23,
-16, -27, 15, 6, 9, 17, 10, 5,
-22, -23, -30, -16, -16, -23, -36, -32,
-33, -28, -22, -43, -5, -32, -20, -41,

-74, -35, -18, -18, -11, 15, 4, -17,
-12, 17, 14, 17, 17, 38, 23, 11,
10, 17, 23, 15, 20, 45, 44, 13,
-8, 22, 24, 27, 26, 33, 26, 3,
-18, -4, 21, 24, 27, 23, 9, -11,
-19, -3, 11, 21, 23, 16, 7, -9,
-27, -11, 4, 13, 14, 4, -5, -17,
-53, -34, -21, -11, -28, -14, -24, -43
};

static inline int getGamePhaseScore(){


int whitePiecesScore = 0 , blackPiecesScore = 0;

for(int piece = N; piece <=Q; piece++){
whitePiecesScore += countBits(pieceBitBoards[piece]) * materialScore[OPENING][piece];
}
for(int piece = n; piece <=q; piece++){
blackPiecesScore += countBits(pieceBitBoards[piece]) * materialScore[OPENING][piece];
}

return whitePiecesScore + blackPiecesScore;
}

static inline int evaluate(){

int gamePhaseScore = getGamePhaseScore();

int gamePhase = -1; if(gamePhaseScore > openingPhaseScore){
gamePhase = OPENING;
} else if(gamePhaseScore < endgamePhaseScore){
gamePhase = ENDGAME;
} else {
gamePhase = MIDDLEGAME;
}

int score = 0;

U64 bitBoardOfPiece;

int pawnsInFileCount = 0;

int piece , square;

for(int consideredPiece = P; consideredPiece <= k; consideredPiece++){

bitBoardOfPiece = pieceBitBoards[consideredPiece];

while(bitBoardOfPiece){

piece = consideredPiece;

square = getLS1BIndex(bitBoardOfPiece);

if (gamePhase == MIDDLEGAME){
score += (
(
( materialScore[OPENING][piece] * 
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
score += materialScore[gamePhase][piece];
}

switch (piece)
{
case P: 

if (gamePhase == MIDDLEGAME){

score += (
(
( positionalScore[OPENING][PAWN][square] * 
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
score += positionalScore[gamePhase][PAWN][square];
}

break;
case N: 
if (gamePhase == MIDDLEGAME){

score += (
(
( positionalScore[OPENING][KNIGHT][square] * 
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
score += positionalScore[gamePhase][KNIGHT][square];
}

break;
case B: 
if (gamePhase == MIDDLEGAME){

score += (
(
( positionalScore[OPENING][BISHOP][square] * 
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
score += positionalScore[gamePhase][BISHOP][square];
}

break;
case Q: 

if (gamePhase == MIDDLEGAME){

score += (
(
( positionalScore[OPENING][QUEEN][square] * 
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
score += positionalScore[gamePhase][QUEEN][square];
}

break;
case R: 
if (gamePhase == MIDDLEGAME){

score += (
(
( positionalScore[OPENING][ROOK][square] * 
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
score += positionalScore[gamePhase][ROOK][square];
}

break;
case K: 
if (gamePhase == MIDDLEGAME){

score += (
(
( positionalScore[OPENING][KING][square] * 
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
score += positionalScore[gamePhase][KING][square];
}

break;

case p: 
if (gamePhase == MIDDLEGAME){

score -= (
(
( positionalScore[OPENING][PAWN][mirrorScope[square]] * 
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
score -= positionalScore[gamePhase][PAWN][mirrorScope[square]];
}

break;
case n: 
if (gamePhase == MIDDLEGAME){

score -= (
(
( positionalScore[OPENING][KNIGHT][mirrorScope[square]] * 
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
score -= positionalScore[gamePhase][KNIGHT][mirrorScope[square]];
}
break;
case b: 
if (gamePhase == MIDDLEGAME){

score -= (
(
( positionalScore[OPENING][BISHOP][mirrorScope[square]] * 
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
score -= positionalScore[gamePhase][BISHOP][mirrorScope[square]];
}

break;
case q: 

if (gamePhase == MIDDLEGAME){

score -= (
(
( positionalScore[OPENING][QUEEN][mirrorScope[square]] * 
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
score -= positionalScore[gamePhase][QUEEN][mirrorScope[square]];
}

break;
case r: 
if (gamePhase == MIDDLEGAME){

score -= (
(
( positionalScore[OPENING][ROOK][mirrorScope[square]] * 
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
score -= positionalScore[gamePhase][ROOK][mirrorScope[square]];
}
break;
case k: 
if (gamePhase == MIDDLEGAME){

score -= (
(
( positionalScore[OPENING][KING][mirrorScope[square]] * 
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
score -= positionalScore[gamePhase][KING][mirrorScope[square]];
}

break;
}

popBitOnSquare(bitBoardOfPiece , square);
}

}
return (sideToMove == WHITE) ? score : (-score); }





static int MVV_LVA[12][12] = {
105, 205, 305, 405, 505, 605, 105, 205, 305, 405, 505, 605,
104, 204, 304, 404, 504, 604, 104, 204, 304, 404, 504, 604,
103, 203, 303, 403, 503, 603, 103, 203, 303, 403, 503, 603,
102, 202, 302, 402, 502, 602, 102, 202, 302, 402, 502, 602,
101, 201, 301, 401, 501, 601, 101, 201, 301, 401, 501, 601,
100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 600,

105, 205, 305, 405, 505, 605, 105, 205, 305, 405, 505, 605,
104, 204, 304, 404, 504, 604, 104, 204, 304, 404, 504, 604,
103, 203, 303, 403, 503, 603, 103, 203, 303, 403, 503, 603,
102, 202, 302, 402, 502, 602, 102, 202, 302, 402, 502, 602,
101, 201, 301, 401, 501, 601, 101, 201, 301, 401, 501, 601,
100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 600
};

static int killerMoves[2][MAX_PLY];

static int historyMoves[12][64];



int pvLength[MAX_PLY];

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

int followPV , scorePV;



U64 transpositionTableEntries = 0;

typedef struct { U64 key; int depth; int flags; int eval; } TranspositionTable; TranspositionTable * transpositionTable = NULL;

void clearTranspositionTable(){

TranspositionTable * hashEntry;

for(hashEntry = transpositionTable; hashEntry < transpositionTable + transpositionTableEntries; hashEntry++){
hashEntry->key = 0;
hashEntry->depth = 0;
hashEntry->flags = 0;
hashEntry->eval = 0;
}
}

void initTranspositionTable(int sizeInMB){
int hashTableSize = 0x100000 * sizeInMB;

transpositionTableEntries = hashTableSize / sizeof(TranspositionTable);

if(transpositionTable != NULL) {
printf("    Clearing hash memory...\n");
free(transpositionTable);
}

transpositionTable = (TranspositionTable *)malloc(sizeof(TranspositionTable) * transpositionTableEntries);

if(transpositionTable == NULL){

printf("    Couldn't allocate memory for hash table, tryinr %dMB...", sizeInMB / 2);
initTranspositionTable(sizeInMB / 2);
} 
else{
clearTranspositionTable();
printf("    Hash table is initialied with %llu entries\n", transpositionTableEntries);
} 
}
static inline int readFromHashTable(int alpha , int beta , int depth){
TranspositionTable * ttPtr = &transpositionTable[hashKey % transpositionTableEntries]; if(ttPtr->key == hashKey){
if(ttPtr->depth >= depth){ 
int eval = ttPtr->eval;
if (eval < (-MATE_SCORE)) eval -= ply;
if (eval > MATE_SCORE) eval += ply;

if(ttPtr->flags == hashPV){
return eval;
}
else if( (ttPtr->flags == hashALPHA) && (eval <= alpha) ){
return alpha;
}
else if( (ttPtr->flags == hashBETA) && (eval >= beta) ){
return beta;
}
}
}
return HASH_NOT_FOUND_EVAL;

}

static inline void writeToHashTable(int evaluation , int depth , int hashFlag){
TranspositionTable * ttPtr = &transpositionTable[hashKey % transpositionTableEntries]; if (evaluation < (-MATE_SCORE)) evaluation -= ply;
if (evaluation > MATE_SCORE) evaluation += ply;
ttPtr->key = hashKey;
ttPtr->depth = depth;
ttPtr->eval = evaluation;
ttPtr->flags = hashFlag;
}



static inline void enablePVScoring(Move * moveList , int depth){

followPV = 0;
for(int count = 0; count < moveList->moveCount; count++){
if(pvTable[0][ply] == moveList->moves[count]){
scorePV = 1;

followPV = 1;
}
}
}

static inline int scoreMove(int move){
if(scorePV){
if(pvTable[0][ply] == move){

scorePV = 0;

return 20000; }
}


if(extractCaptureFlag(move)){
int targetPiece = P;

int startPiece , endPiece; 

if(sideToMove == WHITE){
startPiece = p;
endPiece = k;
}
else{
startPiece = P;
endPiece = K;
}

for(int consideredPiece = startPiece ; consideredPiece <= endPiece ; consideredPiece++){
if(getBitOnSquare(pieceBitBoards[consideredPiece] , extractTarget(move))){
targetPiece = consideredPiece;
break; } 
}
return MVV_LVA[extractMovedPiece(move)][targetPiece] + 10000; }
else{
if(killerMoves[0][ply] == move){
return 9000;
}
else if(killerMoves[1][ply] == move){
return 8000;
}
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

static inline int sortMoves(Move * moveList){
int moveScores[moveList->moveCount];

for(int count = 0; count < moveList->moveCount;count++){
moveScores[count] = scoreMove(moveList->moves[count]);
}
for(int current = 0; current < moveList->moveCount; current++){
for(int next = current+1; next < moveList->moveCount; next++){
if(moveScores[current] < moveScores[next]){
int temp = moveScores[next];
moveScores[next] = moveScores[current];
moveScores[current] = temp;

int tempMove = moveList->moves[next];
moveList->moves[next] = moveList->moves[current];
moveList->moves[current] = tempMove;
}
}
}

}



int quit = 0;

int movesToGo = 40;

int moveTime = -1;

int time = -1;

int inc = 0;

int startTime = 0;

int stopTime = 0;

int timeSet = 0;

int stopped = 0;



int getTimeInMilliSeconds(){
return GetTickCount();
struct timeval timeValue;
gettimeofday(&timeValue , NULL);
return timeValue.tv_sec * 1000 + timeValue.tv_usec / 1000; 
}

int inputWaiting()
{
fd_set readfds;
struct timeval tv;
FD_ZERO (&readfds);
FD_SET (fileno(stdin), &readfds);
tv.tv_sec=0; tv.tv_usec=0;
select(16, &readfds, 0, 0, &tv);

return (FD_ISSET(fileno(stdin), &readfds));
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

}

void readInput()
{
int bytes;

char input[256] = "", *endc;

if (inputWaiting())
{
stopped = 1;

do
{
bytes=read(fileno(stdin), input, 256);
}

while (bytes < 0);

endc = strchr(input,'\n');

if (endc) *endc=0;

if (strlen(input) > 0)
{
if (!strncmp(input, "quit", 4))
{
quit = 1;
}

else if (!strncmp(input, "stop", 4)) {
quit = 1;
}
} 
}
}

static void communicate() {
if(timeSet == 1 && (getTimeInMilliSeconds() > stopTime)) {
stopped = 1;
}

readInput();
}




U64 searchNodes;

static inline int isRepeated(){
for(int idx = 0; idx < repetitionIndex; idx++){
if(repetitionTable[idx] == hashKey){
return 1;
}
}

return 0;
}

static inline int quiescenceSearch(int alpha , int beta){

if((searchNodes & 2047) == 0){
communicate();
}

searchNodes++;

if (ply > (MAX_PLY - 1)) return evaluate();

int evaluation = evaluate();

if (evaluation >= beta) return beta; if (evaluation > alpha){ alpha = evaluation; 
} 
Move moveList[1];

generateMoves(moveList);

sortMoves(moveList);

for(int count = 0; count < moveList->moveCount; count++){

copyBoardState();

ply++;

repetitionIndex++;
repetitionTable[repetitionIndex] = hashKey;

if(makeMove(moveList->moves[count] , captureMoves) == 0){
ply--;

repetitionIndex--;

continue;
}

int score = -quiescenceSearch(-beta , -alpha);

ply--;

repetitionIndex--;

restoreBoardState();

if(stopped == 1) return 0;

if (score > alpha){ alpha = score; 

if (score >= beta)
{
return beta;
} 
} 
}

return alpha;
}

const int fullDepthMoves = 4;
const int reductionLimit = 3;

static inline int negaMaxSearch(int alpha , int beta , int depth){

int score;

int hashFlag = hashALPHA; if (ply && isRepeated()) return 0;

int pvNode = (beta - alpha > 1);

if(ply && ((score = readFromHashTable(alpha , beta , depth)) != HASH_NOT_FOUND_EVAL) && (pvNode == 0)){
return score;
}

pvLength[ply] = ply;

if((searchNodes & 2047) == 0){
communicate();
}

if (depth == 0){
return quiescenceSearch(alpha , beta);
}

if (ply > (MAX_PLY - 1)) return evaluate();

searchNodes++; int isCheck = isSquareAttacked((sideToMove == WHITE) ? getLS1BIndex(pieceBitBoards[K]) : getLS1BIndex(pieceBitBoards[k]) , sideToMove ^ 1);

if(isCheck) depth++;

int legalMoves = 0;

if ((depth >= 3) && (isCheck == 0) && (ply)){

copyBoardState();

ply++;

repetitionIndex++;
repetitionTable[repetitionIndex] = hashKey;

if(enPassant != NULL_SQUARE){
hashKey ^= enPassantKeys[enPassant];
}

enPassant = NULL_SQUARE;

sideToMove ^= 1;

hashKey ^= sideToMoveKey;

score = -negaMaxSearch(-beta , -beta + 1 , depth - 1 - 2); ply--;

repetitionIndex--;

restoreBoardState();

if (stopped == 1) return 0;

if(score >= beta) return beta;

}

Move moveList[1];

generateMoves(moveList);

if(followPV) enablePVScoring(moveList , depth);

sortMoves(moveList);

int movesSearched = 0;

for(int count = 0; count < moveList->moveCount; count++){

copyBoardState();

ply++;

repetitionIndex++;
repetitionTable[repetitionIndex] = hashKey;

if(makeMove(moveList->moves[count] , allMoves) == 0){
ply--;

repetitionIndex--;

continue;
}

legalMoves++;


if(movesSearched == 0){
score = -negaMaxSearch(-beta , -alpha , depth-1);
}
else{
if(
(movesSearched >= fullDepthMoves) 
&& (depth >= reductionLimit) 
&& (isCheck == 0)
&& (extractCaptureFlag(moveList->moves[count]) == 0)
&& (extractPromotedPiece(moveList->moves[count]) == 0)
){
score = -negaMaxSearch(-alpha - 1 , -alpha , depth - 2 );
}
else{
score = alpha + 1; }
if(score > alpha){
score = -negaMaxSearch(-alpha - 1 , -alpha , depth - 1 );
if((score > alpha) && (score < beta)){
score = -negaMaxSearch(-beta , -alpha , depth - 1 );
}
}
}
ply--;

repetitionIndex--;

restoreBoardState();

if (stopped == 1) return 0;

movesSearched++;

if (score > alpha){ hashFlag = hashPV;

if(extractCaptureFlag(moveList->moves[count]) == 0){
historyMoves[extractMovedPiece(moveList->moves[count])][extractTarget(moveList->moves[count])] += depth;
}

alpha = score; pvTable[ply][ply] = moveList->moves[count];

for(int nextPly = ply+1 ; nextPly < pvLength[ply+1] ; nextPly++){
pvTable[ply][nextPly] = pvTable[ply+1][nextPly];
}

pvLength[ply] = pvLength[ply+1];

if (score >= beta){
writeToHashTable(beta , depth , hashBETA);

if(extractCaptureFlag(moveList->moves[count]) == 0){
killerMoves[1][ply] = killerMoves[0][ply];
killerMoves[0][ply] = moveList->moves[count];
}

return beta; } 

} 
}

if(legalMoves == 0){
if(isCheck){
return -MATE_VALUE + ply;
}
else{
return 0;
}

}

writeToHashTable(alpha , depth , hashFlag); return alpha;

}


void searchPosition(int depth){

int score = 0;

searchNodes = 0;

stopped = 0;

followPV = 0;
scorePV = 0;

memset(killerMoves , 0 , sizeof(killerMoves));
memset(historyMoves , 0 , sizeof(historyMoves));
memset(pvTable , 0 , sizeof(pvTable));
memset(pvLength , 0 , sizeof(pvLength));


int alpha = -INFINITY; 
int beta = INFINITY;


for(int currDepth = 1; currDepth <= depth; currDepth++){

if (stopped == 1) {
break;
}

followPV = 1;

score = negaMaxSearch(alpha , beta , currDepth);

if ((score <= alpha) || (score >= beta)){
alpha = -INFINITY; 
beta = INFINITY;
continue;
}
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
printMove(pvTable[0][count]);
printf(" ");
}
printf("\n");
}

printf("bestmove ");
printMove(pvTable[0][0]);
printf("\n");
}



int parseMoveString(char * moveString){

Move moveList[1];

generateMoves(moveList);

int srcSquare = 8 * (8 - (moveString[1] - '0')) + (moveString[0] - 'a');
int targetSquare = 8 * (8 - (moveString[3] - '0')) + (moveString[2] - 'a');

for(int count = 0; count < moveList->moveCount; count++){
int move = moveList->moves[count];
if((srcSquare == extractSource(move)) && (targetSquare == extractTarget(move))){
int promotedPiece = extractPromotedPiece(move);
if (promotedPiece){
if ((promotedPiece == Q || promotedPiece == q) && moveString[4] == 'q'){
return move;
}
else if ((promotedPiece == R || promotedPiece == r) && moveString[4] == 'r'){
return move;
}
else if ((promotedPiece == B || promotedPiece == b) && moveString[4] == 'b'){
return move;
}
else if ((promotedPiece == N || promotedPiece == n) && moveString[4] == 'n'){
return move;
}

continue; }
return move;
}
}

return 0;
}

void parsePosition(char * command){

command += 9;

char * currCmd = command;

if(strncmp(command , "startpos" , 8) == 0){
parseFENString(START_POSITION);
}

else{
currCmd = strstr(command , "fen");
if(currCmd == NULL){
parseFENString(START_POSITION);
}
else{
currCmd += 4; parseFENString(currCmd);
}
}

currCmd = strstr(command , "moves");

if(currCmd != NULL){

currCmd += 6;

while(*currCmd){

int move = parseMoveString(currCmd);

if (move == 0) break;

repetitionIndex++;
repetitionTable[repetitionIndex] = hashKey;

makeMove(move , allMoves);

while(*currCmd && *currCmd != ' ') currCmd++;
currCmd++;

}
}

printBoard();
}

void resetTimeControl(){
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

void parseGo(char *command)
{
resetTimeControl();

int depth = -1;

char *argument = NULL;

if ((argument = strstr(command,"infinite"))) {}

if ((argument = strstr(command,"binc")) && (sideToMove == BLACK))
inc = atoi(argument + 5);

if ((argument = strstr(command,"winc")) && (sideToMove == WHITE))
inc = atoi(argument + 5);

if ((argument = strstr(command,"wtime")) && (sideToMove == WHITE))
time = atoi(argument + 6);

if ((argument = strstr(command,"btime")) && (sideToMove == BLACK))
time = atoi(argument + 6);

if ((argument = strstr(command,"movestogo")))
movesToGo = atoi(argument + 10);

if ((argument = strstr(command,"movetime")))
movesToGo = atoi(argument + 9);

if ((argument = strstr(command,"depth")))
depth = atoi(argument + 6);

if(moveTime != -1)
{
time = moveTime;

movesToGo = 1;
}

startTime = getTimeInMilliSeconds();

depth = depth;

if(time != -1)
{
timeSet = 1;

time /= movesToGo;

if (time > 1500) time -= 50;
time -= 50;
stopTime = startTime + time + inc;
}

if(depth == -1)
depth = 64;

printf("time:%d start:%u stop:%u depth:%d timeset:%d\n",
time, startTime, stopTime, depth, timeSet);

searchPosition(depth);
}

void mainUCILoop(){

int maxHash = 128;

int mb = 64;
setbuf(stdin , NULL);
setbuf(stdout , NULL);

char input[2000];

printf("ID Name : BBC %s\n" , VERSION);
printf("ID Author : Ashvin Ganesh\n");
printf("option name Hash type spin default 64 min 4 max %d\n" , maxHash);
printf("uciok\n");

while(1){
memset(input , 0 , sizeof(input));

fflush(stdout);

if(!(fgets(input , 2000 , stdin))) continue;
if(input[0] == '\n') continue;

if (strncmp(input , "isready" , 7) == 0){
printf("readyok\n");
continue;
}
else if (strncmp(input , "position" , 8) == 0){
parsePosition(input);
clearTranspositionTable();
}
else if (strncmp(input , "ucinewgame" , 10) == 0){
parsePosition("position startpos");
clearTranspositionTable();
}
else if (strncmp(input , "go" , 2) == 0){
parseGo(input);
}
else if (strncmp(input , "quit" , 4) == 0){
break;
}
else if (strncmp(input , "uci" , 3) == 0){
printf("ID Name : BBC\n");
printf("ID Name : Ashvin Ganesh\n");
printf("uciok\n");
}
else if(!(strncmp(input , "setoption name Hash value" , 26))){
sscanf(input, "%*s %*s %*s %*s %d" , &mb);

if(mb < 4) mb = 4;
if(mb > maxHash) mb = maxHash;

printf("Set hash table size to %dMB\n" , mb);
initTranspositionTable(mb);
}

}
}



U64 nodes;

static inline void perftDriver(int depth){
if (depth == 0) {
nodes++;
return;
}

Move moveList[1];
generateMoves(moveList);

for(int count = 0; count < moveList->moveCount; count++){
copyBoardState();

if (!makeMove(moveList->moves[count] , allMoves)) continue; perftDriver(depth - 1);
restoreBoardState();

}
}

static inline void perftTest(int depth){

printf(" Performance Test \n");

Move moveList[1];
generateMoves(moveList);

int start = getTimeInMilliSeconds();

for(int count = 0; count < moveList->moveCount; count++){
copyBoardState();

if (!makeMove(moveList->moves[count] , allMoves)) continue; long cummulativeNodes = nodes;

perftDriver(depth - 1);
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



void init(){
generatePawnAttacks();
generateKnightAttacks();
generateKingAttacks();
initBishopAndRookAttacks(bishop);
initBishopAndRookAttacks(rook);
initRandomKeys(); initTranspositionTable(12); initEvaluationMasks();
}



int main(){
init();
int debug = 0;

if(debug){
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
