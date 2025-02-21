#include "Symbol_Table.h"

// Token types
typedef enum
{
    KEYWORD,
    IDENTIFIER,
    OPERATOR,
    LITERAL_STRING,
    LITERAL_NUMBER,
    SEPARATOR,
    UNKNOWN
} TokenType;

// Token structure
typedef struct
{
    char tokenString[100];
    TokenType type;
    int row;
    int col;
} Token;

// List of C keywords for checking
const char *keywords[] = {
    "auto", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "int", "long", "register", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while"};

const int NUM_KEYWORDS = sizeof(keywords) / sizeof(keywords[0]);

// Function to check if a string is a keyword
int isKeyword(const char *str)
{
    for (int i = 0; i < NUM_KEYWORDS; i++)
    {
        if (strcmp(str, keywords[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

// Function to check if character is operator
int isOperator(char c)
{
    return (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' ||
            c == '=' || c == '<' || c == '>' || c == '!' || c == '&' ||
            c == '|' || c == '^' || c == '~' || c == '?' || c == ':');
}

// Function to check if character is separator
int isSeparator(char c)
{
    return (c == '(' || c == ')' || c == '{' || c == '}' || c == '[' ||
            c == ']' || c == ';' || c == ',' || c == '.');
}

void processDeclaration(Token token)
{
    struct token tk;
    tk.lexeme = strdup(token.tokenString); // Duplicate string for storage
    tk.rowno = token.row;
    tk.colno = token.col;

    // Check if it's a function (simple check, extend this logic if needed)
    if (strchr(token.tokenString, '(') != NULL)
    {
        // It's a function (we assume parentheses for functions)
        
        // Try to determine the return type of the function (before the function name)
        // For simplicity, we check for known return types. 
        // If no type is found, we mark it as void with size 0.
        const char* returnType = "void";  // Default to void
        int returnTypeFound = 0;

        // Check for known return types like int, char, etc. (This can be expanded)
        if (strcmp(token.tokenString, "int") == 0) {
            returnType = "int";
            returnTypeFound = 1;
        } else if (strcmp(token.tokenString, "char") == 0) {
            returnType = "char";
            returnTypeFound = 1;
        } else if (strcmp(token.tokenString, "float") == 0) {
            returnType = "float";
            returnTypeFound = 1;
        } else if (strcmp(token.tokenString, "double") == 0) {
            returnType = "double";
            returnTypeFound = 1;
        }

        // Assign function type based on return type
        if (returnTypeFound) {
            tk.type = strcmp(returnType, "int") == 0 ? INT :
                      strcmp(returnType, "char") == 0 ? CHAR :
                      strcmp(returnType, "float") == 0 ? FLOAT :
                      strcmp(returnType, "double") == 0 ? DOUBLE : FUNC;
            tk.index = HASH(tk.lexeme); // Assign an index for hashing
        } else {
            // If return type is not found, mark as void
            tk.type = FUNC;  // Mark as function type
            tk.index = HASH(tk.lexeme); // Assign an index for hashing
        }

        // Insert the function declaration into the symbol table
        INSERT(tk);
        return;
    }

    // For other declarations, handle them as variables (only treat as variables if it's a type)
    if (strcmp(token.tokenString, "int") == 0)
    {
        tk.type = INT;
        INSERT(tk);
        return;
    }
    else if (strcmp(token.tokenString, "float") == 0)
    {
        tk.type = FLOAT;
        INSERT(tk);
        return;
    }
    else if (strcmp(token.tokenString, "double") == 0)
    {
        tk.type = DOUBLE;
        INSERT(tk);
        return;
    }
    else if (strcmp(token.tokenString, "char") == 0)
    {
        tk.type = CHAR;
        INSERT(tk);
        return;
    }

    // If it's not a recognized type, treat it as an unknown identifier
    tk.type = UNKNOWN;
    INSERT(tk);
}

Token getIdentifierOrKeyword(FILE* fa, int row, int col) {
    Token token = {.row = row, .col = col};
    int idx = 0;
    char c;

    // Read the token until a non-alphanumeric character is encountered
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

    // If it's a keyword, we simply return the token and do nothing more
    if (token.type == KEYWORD) {
        return token; // No insertion into symbol table for keywords
    }
    // If it's an identifier, process it for symbol table insertion
    else if (token.type == IDENTIFIER) {
        processDeclaration(token); // Handle the identifier processing (variables/functions)
    }

    return token;
}

// Function to get numeric literal
Token getNumber(FILE *fa, int row, int col)
{
    Token token = {.row = row, .col = col, .type = LITERAL_NUMBER};
    int idx = 0;
    char c;

    while ((c = getc(fa)) != EOF && (isdigit(c) || c == '.'))
    {
        if (idx < 99)
        {
            token.tokenString[idx++] = c;
        }
    }
    token.tokenString[idx] = '\0';

    if (c != EOF)
    {
        ungetc(c, fa);
    }

    return token;
}

// Function to get string literal
Token getString(FILE *fa, int row, int col)
{
    Token token = {.row = row, .col = col, .type = LITERAL_STRING};
    int idx = 0;
    char c;

    c = getc(fa); // Consume the first quote
    token.tokenString[idx++] = '"';

    while ((c = getc(fa)) != EOF && c != '"')
    {
        if (idx < 99)
        {
            token.tokenString[idx++] = c;
        }
    }

    if (c == '"')
    { // If closing quote is found, append it
        token.tokenString[idx++] = '"';
    }

    token.tokenString[idx] = '\0';

    return token;
}

// Function to get operator
Token getOperator(FILE *fa, int row, int col)
{
    Token token = {.row = row, .col = col, .type = OPERATOR};
    char c = getc(fa);
    int idx = 0;

    token.tokenString[idx++] = c;

    // Check for multi-character operators
    char next = getc(fa);
    if ((c == '+' && next == '+') || (c == '-' && next == '-') ||
        (c == '=' && next == '=') || (c == '!' && next == '=') ||
        (c == '<' && next == '=') || (c == '>' && next == '=') ||
        (c == '&' && next == '&') || (c == '|' && next == '|'))
    {
        token.tokenString[idx++] = next;
    }
    else
    {
        ungetc(next, fa);
    }

    token.tokenString[idx] = '\0';
    return token;
}

// Function to get separator
Token getSeparator(FILE *fa, int row, int col)
{
    Token token = {.row = row, .col = col, .type = SEPARATOR};
    char c = getc(fa);
    token.tokenString[0] = c;
    token.tokenString[1] = '\0';
    return token;
}

// Function to print token
void printToken(Token token)
{
    const char *typeStr;
    switch (token.type)
    {
    case KEYWORD:
        typeStr = "KEYWORD";
        break;
    case IDENTIFIER:
        typeStr = "IDENTIFIER";
        break;
    case OPERATOR:
        typeStr = "OPERATOR";
        break;
    case LITERAL_STRING:
        typeStr = "STRING";
        break;
    case LITERAL_NUMBER:
        typeStr = "NUMBER";
        break;
    case SEPARATOR:
        typeStr = "SEPARATOR";
        break;
    default:
        typeStr = "UNKNOWN";
        break;
    }
    printf("<%s, %s, row:%d, col:%d>\n", token.tokenString, typeStr, token.row, token.col);
}

// Main tokenizer function
void tokenize(FILE *fa)
{
    int row = 1, col = 1;
    char c;

    while ((c = getc(fa)) != EOF)
    {
        if (c == '\n')
        {
            row++;
            col = 1;
            continue;
        }

        if (isspace(c))
        {
            col++;
            continue;
        }

        ungetc(c, fa);
        Token token;

        if (isalpha(c) || c == '_')
        {
            token = getIdentifierOrKeyword(fa, row, col);
            col += strlen(token.tokenString);
        }
        else if (isdigit(c))
        {
            token = getNumber(fa, row, col); // Ensure this function is defined above or included.
            col += strlen(token.tokenString);
        }
        else if (c == '"')
        {
            token = getString(fa, row, col); // Ensure this function is defined above or included.
            col += strlen(token.tokenString);
        }
        else if (isOperator(c))
        {
            token = getOperator(fa, row, col); // Ensure this function is defined above or included.
            col += strlen(token.tokenString);
        }
        else if (isSeparator(c))
        {
            token = getSeparator(fa, row, col); // Ensure this function is defined above or included.
            col += strlen(token.tokenString);
        }
        else
        {
            // Skip unknown character
            getc(fa);
            col++;
            continue;
        }

        printToken(token); // Ensure this function is defined above or included.
    }
}

int main()
{
    FILE *fa = fopen("../Lexical Analyser/preprocessed.c", "r");
    if (fa == NULL)
    {
        printf("Cannot open file\n");
        return 1;
    }
    tokenize(fa);
    Display();
    fclose(fa);
    return 0;
}
