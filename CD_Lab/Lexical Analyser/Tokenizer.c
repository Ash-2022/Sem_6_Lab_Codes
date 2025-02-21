#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

// Token types
typedef enum {
    KEYWORD,
    IDENTIFIER,
    OPERATOR,
    LITERAL_STRING,
    LITERAL_NUMBER,
    SEPARATOR,
    UNKNOWN
} TokenType;

// Token structure
typedef struct {
    char tokenString[100];
    TokenType type;
    int row;
    int col;
} Token;

// List of C keywords for checking
const char* keywords[] = {
    "auto", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "int", "long", "register", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while"
};

const int NUM_KEYWORDS = sizeof(keywords) / sizeof(keywords[0]);

// Function to check if a string is a keyword
int isKeyword(const char* str) {
    for (int i = 0; i < NUM_KEYWORDS; i++) {
        if (strcmp(str, keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Function to check if character is operator
int isOperator(char c) {
    return (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' ||
            c == '=' || c == '<' || c == '>' || c == '!' || c == '&' ||
            c == '|' || c == '^' || c == '~' || c == '?' || c == ':');
}

// Function to check if character is separator
int isSeparator(char c) {
    return (c == '(' || c == ')' || c == '{' || c == '}' || c == '[' ||
            c == ']' || c == ';' || c == ',' || c == '.');
}

// Function to get identifier or keyword
Token getIdentifierOrKeyword(FILE* fa, int row, int col) {
    Token token = {.row = row, .col = col};
    int idx = 0;
    char c;

    while ((c = getc(fa)) != EOF && (isalnum(c) || c == '_')) {
        if (idx < 99) {
            token.tokenString[idx++] = c;
        }
    }
    token.tokenString[idx] = '\0';

    if (c != EOF) {
        ungetc(c, fa);
    }

    // Determine if it's a keyword or identifier
    token.type = isKeyword(token.tokenString) ? KEYWORD : IDENTIFIER;

    return token;
}

// Function to get numeric literal
Token getNumber(FILE* fa, int row, int col) {
    Token token = {.row = row, .col = col, .type = LITERAL_NUMBER};
    int idx = 0;
    char c;

    while ((c = getc(fa)) != EOF && (isdigit(c) || c == '.')) {
        if (idx < 99) {
            token.tokenString[idx++] = c;
        }
    }
    token.tokenString[idx] = '\0';

    if (c != EOF) {
        ungetc(c, fa);
    }

    return token;
}

// Function to get string literal 
Token getString(FILE* fa, int row, int col) {
    Token token = {.row = row, .col = col, .type = LITERAL_STRING};
    int idx = 0;
    char c;

    c = getc(fa);  // Consume the first quote
    token.tokenString[idx++] = '"';

    while ((c = getc(fa)) != EOF && c != '"') {
        if (idx < 99) {
            token.tokenString[idx++] = c;
        }
    }

    if (c == '"') { // If closing quote is found, append it
        token.tokenString[idx++] = '"';
    }

    token.tokenString[idx] = '\0';

    return token;
}


// Function to get operator
Token getOperator(FILE* fa, int row, int col) {
    Token token = {.row = row, .col = col, .type = OPERATOR};
    char c = getc(fa);
    int idx = 0;

    token.tokenString[idx++] = c;
    
    // Check for multi-character operators
    char next = getc(fa);
    if ((c == '+' && next == '+') || (c == '-' && next == '-') ||
        (c == '=' && next == '=') || (c == '!' && next == '=') ||
        (c == '<' && next == '=') || (c == '>' && next == '=') ||
        (c == '&' && next == '&') || (c == '|' && next == '|')) {
        token.tokenString[idx++] = next;
    } else {
        ungetc(next, fa);
    }
    
    token.tokenString[idx] = '\0';
    return token;
}

// Function to get separator
Token getSeparator(FILE* fa, int row, int col) {
    Token token = {.row = row, .col = col, .type = SEPARATOR};
    char c = getc(fa);
    token.tokenString[0] = c;
    token.tokenString[1] = '\0';
    return token;
}

// Function to print token
void printToken(Token token) {
    const char* typeStr;
    switch (token.type) {
        case KEYWORD: typeStr = "KEYWORD"; break;
        case IDENTIFIER: typeStr = "IDENTIFIER"; break;
        case OPERATOR: typeStr = "OPERATOR"; break;
        case LITERAL_STRING: typeStr = "STRING"; break;
        case LITERAL_NUMBER: typeStr = "NUMBER"; break;
        case SEPARATOR: typeStr = "SEPARATOR"; break;
        default: typeStr = "UNKNOWN"; break;
    }
    printf("<%s, %s, row:%d, col:%d>\n", token.tokenString, typeStr, token.row, token.col);
}

// Main tokenizer function
void tokenize(FILE* fa) {
    int row = 1, col = 1;
    char c;

    while ((c = getc(fa)) != EOF) {
        if (c == '\n') {
            row++;
            col = 1;
            continue;
        }
        
        if (isspace(c)) {
            col++;
            continue;
        }

        ungetc(c, fa);
        Token token;

        if (isalpha(c) || c == '_') {
            token = getIdentifierOrKeyword(fa, row, col);
            col += strlen(token.tokenString);
        }
        else if (isdigit(c)) {
            token = getNumber(fa, row, col);
            col += strlen(token.tokenString);
        }
        else if (c == '"') {
            token = getString(fa, row, col);
            col += strlen(token.tokenString);
        }
        else if (isOperator(c)) {
            token = getOperator(fa, row, col);
            col += strlen(token.tokenString);
        }
        else if (isSeparator(c)) {
            token = getSeparator(fa, row, col);
            col += strlen(token.tokenString);
        }
        else {
            // Skip unknown character
            getc(fa);
            col++;
            continue;
        }

        printToken(token);
    }
}

int main() {
    FILE* fa = fopen("preprocessed.c", "r");
    if (fa == NULL) {
        printf("Cannot open file\n");
        return 1;
    }

    tokenize(fa);
    fclose(fa);
    return 0;
}
