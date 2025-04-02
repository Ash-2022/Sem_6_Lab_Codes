#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100
#define MAX_BUFFER_SIZE 1000

typedef enum {
    TOKEN_MAIN,
    TOKEN_INT,
    TOKEN_CHAR,
    TOKEN_ID,
    TOKEN_NUM,
    TOKEN_SEMICOLON,
    TOKEN_COMMA,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_ASSIGN,
    TOKEN_EOF,
    TOKEN_UNKNOWN
} TokenType;

typedef struct {
    TokenType type;
    char lexeme[MAX_TOKEN_LEN];
    int row;
    int col;
} Token;

// Global variables
Token currentToken;
FILE *sourceFile;
char buffer[MAX_BUFFER_SIZE];
int bufferIndex = 0;
int lineNum = 1;
int colNum = 0;
int errors = 0;

// Function prototypes
void getNextToken();
void match(TokenType expectedType);
void parseProgram();
void parseDeclarations();
void parseDataType();
void parseIdentifierList();
void parseAssignStat();
void syntaxError(const char *message);

// Utility function to check if a string is a keyword
int isKeyword(const char *str) {
    if (strcmp(str, "main") == 0) return TOKEN_MAIN;
    else if (strcmp(str, "int") == 0) return TOKEN_INT;
    else if (strcmp(str, "char") == 0) return TOKEN_CHAR;
    return -1;
}

// Read the next character from the source file
int getNextChar() {
    int c = fgetc(sourceFile);
    if (c == '\n') {
        lineNum++;
        colNum = 0;
    } else {
        colNum++;
    }
    return c;
}

// Unget a character and update column
void ungetNextChar(int c) {
    ungetc(c, sourceFile);
    colNum--;
}

// Get the next token from the source file
void getNextToken() {
    int c;
    int tokenCol;
    
    // Skip whitespace
    do {
        c = getNextChar();
    } while (c == ' ' || c == '\t' || c == '\n');
    
    if (c == EOF) {
        currentToken.type = TOKEN_EOF;
        strcpy(currentToken.lexeme, "EOF");
        currentToken.row = lineNum;
        currentToken.col = colNum;
        return;
    }
    
    tokenCol = colNum;  // Save starting column
    
    // Identifiers and keywords
    if (isalpha(c) || c == '_') {
        int i = 0;
        do {
            currentToken.lexeme[i++] = c;
            c = getNextChar();
        } while (isalnum(c) || c == '_');
        
        currentToken.lexeme[i] = '\0';
        ungetNextChar(c);
        
        int keywordType = isKeyword(currentToken.lexeme);
        if (keywordType != -1) {
            currentToken.type = keywordType;
        } else {
            currentToken.type = TOKEN_ID;
        }
    }
    // Numbers
    else if (isdigit(c)) {
        int i = 0;
        do {
            currentToken.lexeme[i++] = c;
            c = getNextChar();
        } while (isdigit(c));
        
        currentToken.lexeme[i] = '\0';
        ungetNextChar(c);
        currentToken.type = TOKEN_NUM;
    }
    // Special characters
    else {
        switch (c) {
            case ';':
                currentToken.type = TOKEN_SEMICOLON;
                strcpy(currentToken.lexeme, ";");
                break;
            case ',':
                currentToken.type = TOKEN_COMMA;
                strcpy(currentToken.lexeme, ",");
                break;
            case '(':
                currentToken.type = TOKEN_LPAREN;
                strcpy(currentToken.lexeme, "(");
                break;
            case ')':
                currentToken.type = TOKEN_RPAREN;
                strcpy(currentToken.lexeme, ")");
                break;
            case '{':
                currentToken.type = TOKEN_LBRACE;
                strcpy(currentToken.lexeme, "{");
                break;
            case '}':
                currentToken.type = TOKEN_RBRACE;
                strcpy(currentToken.lexeme, "}");
                break;
            case '=':
                currentToken.type = TOKEN_ASSIGN;
                strcpy(currentToken.lexeme, "=");
                break;
            default:
                currentToken.type = TOKEN_UNKNOWN;
                currentToken.lexeme[0] = c;
                currentToken.lexeme[1] = '\0';
        }
    }
    
    currentToken.row = lineNum;
    currentToken.col = tokenCol;
}

// Print an error message with line and column information
void syntaxError(const char *message) {
    errors++;
    fprintf(stderr, "Syntax Error at line %d, col %d: %s. Found '%s'\n", 
            currentToken.row, currentToken.col, message, currentToken.lexeme);
}

// Match the current token with the expected type
void match(TokenType expectedType) {
    if (currentToken.type == expectedType) {
        getNextToken();
    } else {
        char expected[50];
        switch (expectedType) {
            case TOKEN_MAIN: strcpy(expected, "main"); break;
            case TOKEN_INT: strcpy(expected, "int"); break;
            case TOKEN_CHAR: strcpy(expected, "char"); break;
            case TOKEN_ID: strcpy(expected, "identifier"); break;
            case TOKEN_NUM: strcpy(expected, "number"); break;
            case TOKEN_SEMICOLON: strcpy(expected, ";"); break;
            case TOKEN_COMMA: strcpy(expected, ","); break;
            case TOKEN_LPAREN: strcpy(expected, "("); break;
            case TOKEN_RPAREN: strcpy(expected, ")"); break;
            case TOKEN_LBRACE: strcpy(expected, "{"); break;
            case TOKEN_RBRACE: strcpy(expected, "}"); break;
            case TOKEN_ASSIGN: strcpy(expected, "="); break;
            default: strcpy(expected, "unknown token");
        }
        
        syntaxError(expected);
        
        // Simple error recovery: skip to the next token
        getNextToken();
    }
}

// Program → main() { declarations assign_stat }
void parseProgram() {
    if (currentToken.type == TOKEN_MAIN) {
        match(TOKEN_MAIN);
        match(TOKEN_LPAREN);
        match(TOKEN_RPAREN);
        match(TOKEN_LBRACE);
        parseDeclarations();
        parseAssignStat();
        match(TOKEN_RBRACE);
    } else {
        syntaxError("Expected 'main'");
    }
}

// declarations → data-type identifier-list; declarations | ε
void parseDeclarations() {
    if (currentToken.type == TOKEN_INT || currentToken.type == TOKEN_CHAR) {
        parseDataType();
        parseIdentifierList();
        match(TOKEN_SEMICOLON);
        parseDeclarations();
    }
    // ε production, do nothing and exit the function
}

// data-type → int | char
void parseDataType() {
    if (currentToken.type == TOKEN_INT) {
        match(TOKEN_INT);
    } else if (currentToken.type == TOKEN_CHAR) {
        match(TOKEN_CHAR);
    } else {
        syntaxError("Expected 'int' or 'char'");
    }
}

// identifier-list → id | id, identifier-list
void parseIdentifierList() {
    if (currentToken.type == TOKEN_ID) {
        match(TOKEN_ID);
        if (currentToken.type == TOKEN_COMMA) {
            match(TOKEN_COMMA);
            parseIdentifierList();
        }
    } else {
        syntaxError("Expected identifier");
    }
}

// assign_stat → id=id; | id=num;
void parseAssignStat() {
    if (currentToken.type == TOKEN_ID) {
        match(TOKEN_ID);
        match(TOKEN_ASSIGN);
        
        if (currentToken.type == TOKEN_ID) {
            match(TOKEN_ID);
        } else if (currentToken.type == TOKEN_NUM) {
            match(TOKEN_NUM);
        } else {
            syntaxError("Expected identifier or number after '='");
        }
        
        match(TOKEN_SEMICOLON);
    } else {
        syntaxError("Expected identifier for assignment");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <source_file>\n", argv[0]);
        return 1;
    }
    
    sourceFile = fopen(argv[1], "r");
    if (!sourceFile) {
        fprintf(stderr, "Error opening file: %s\n", argv[1]);
        return 1;
    }
    
    printf("Parsing file: %s\n", argv[1]);
    
    // Initialize and start parsing
    getNextToken();
    parseProgram();
    
    if (errors == 0) {
        printf("Parsing completed successfully.\n");
    } else {
        printf("Parsing completed with %d errors.\n", errors);
    }
    
    fclose(sourceFile);
    return 0;
}
