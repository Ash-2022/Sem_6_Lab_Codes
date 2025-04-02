#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_STACK_SIZE 100
#define MAX_INPUT_SIZE 100

// Token types
typedef enum {
    TOKEN_ID,
    TOKEN_PLUS,
    TOKEN_STAR,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_END,
    TOKEN_ERROR
} TokenType;

// Token structure
typedef struct {
    TokenType type;
    char lexeme[20];
} Token;
        
// Stack element structure
typedef struct {
    int state;
    char symbol; // Using char to represent both terminals and non-terminals
} StackElement;

// Global variables
StackElement stack[MAX_STACK_SIZE];
int top = -1;
Token input[MAX_INPUT_SIZE];
int input_size = 0;

// Action table: (state, terminal) -> (action, value)
// Actions: 's' = shift, 'r' = reduce, 'a' = accept, 'e' = error
// For shift: value = next state
// For reduce: value = production number
typedef struct {
    char action;
    int value;
} ActionEntry;

// Action table
ActionEntry action_table[12][6] = {
    /* State 0 */ { {'s',5}, {'e',0}, {'e',0}, {'s',4}, {'e',0}, {'e',0} },  // id, +, *, (, ), $
    /* State 1 */ { {'e',0}, {'s',6}, {'e',0}, {'e',0}, {'e',0}, {'a',0} },  // id, +, *, (, ), $
    /* State 2 */ { {'e',0}, {'r',2}, {'s',7}, {'e',0}, {'r',2}, {'r',2} },  // id, +, *, (, ), $
    /* State 3 */ { {'e',0}, {'r',4}, {'r',4}, {'e',0}, {'r',4}, {'r',4} },  // id, +, *, (, ), $
    /* State 4 */ { {'s',5}, {'e',0}, {'e',0}, {'s',4}, {'e',0}, {'e',0} },  // id, +, *, (, ), $
    /* State 5 */ { {'e',0}, {'r',6}, {'r',6}, {'e',0}, {'r',6}, {'r',6} },  // id, +, *, (, ), $
    /* State 6 */ { {'s',5}, {'e',0}, {'e',0}, {'s',4}, {'e',0}, {'e',0} },  // id, +, *, (, ), $
    /* State 7 */ { {'s',5}, {'e',0}, {'e',0}, {'s',4}, {'e',0}, {'e',0} },  // id, +, *, (, ), $
    /* State 8 */ { {'e',0}, {'s',6}, {'e',0}, {'e',0}, {'s',11}, {'e',0} }, // id, +, *, (, ), $
    /* State 9 */ { {'e',0}, {'r',1}, {'s',7}, {'e',0}, {'r',1}, {'r',1} },  // id, +, *, (, ), $
    /* State 10 */ { {'e',0}, {'r',3}, {'r',3}, {'e',0}, {'r',3}, {'r',3} }, // id, +, *, (, ), $
    /* State 11 */ { {'e',0}, {'r',5}, {'r',5}, {'e',0}, {'r',5}, {'r',5} }  // id, +, *, (, ), $
};

// Goto table: (state, non-terminal) -> (next state)
// Non-terminals: 'E', 'T', 'F'
int goto_table[12][3] = {
    /* State 0 */  {1, 2, 3},
    /* State 1 */  {0, 0, 0},
    /* State 2 */  {0, 0, 0},
    /* State 3 */  {0, 0, 0},
    /* State 4 */  {8, 2, 3},
    /* State 5 */  {0, 0, 0},
    /* State 6 */  {0, 9, 3},
    /* State 7 */  {0, 0, 10},
    /* State 8 */  {0, 0, 0},
    /* State 9 */  {0, 0, 0},
    /* State 10 */ {0, 0, 0},
    /* State 11 */ {0, 0, 0}
};

// Grammar productions
struct {
    char lhs;
    char rhs[10];
    int rhs_len;
} productions[7] = {
    {' ', "", 0},                // Dummy production for indexing
    {'E', "E+T", 3},             // E -> E+T
    {'E', "T", 1},               // E -> T
    {'T', "T*F", 3},             // T -> T*F
    {'T', "F", 1},               // T -> F
    {'F', "(E)", 3},             // F -> (E)
    {'F', "id", 2}               // F -> id (Note: 'id' is treated as 2 chars for stack purposes)
};

// Push to stack
void push(int state, char symbol) {
    if (top >= MAX_STACK_SIZE - 1) {
        printf("Stack overflow\n");
        exit(1);
    }
    top++;
    stack[top].state = state;
    stack[top].symbol = symbol;
}

// Pop from stack
void pop() {
    if (top < 0) {
        printf("Stack underflow\n");
        exit(1);
    }
    top--;
}

// Lexical analyzer to tokenize input
void tokenize(char *input_str) {
    int i = 0;
    int token_idx = 0;
    
    while (input_str[i] != '\0') {
        if (isspace(input_str[i])) {
            i++;
            continue;
        }
        
        Token token;
        
        if (input_str[i] == '+') {
            token.type = TOKEN_PLUS;
            strcpy(token.lexeme, "+");
            i++;
        } else if (input_str[i] == '*') {
            token.type = TOKEN_STAR;
            strcpy(token.lexeme, "*");
            i++;
        } else if (input_str[i] == '(') {
            token.type = TOKEN_LPAREN;
            strcpy(token.lexeme, "(");
            i++;
        } else if (input_str[i] == ')') {
            token.type = TOKEN_RPAREN;
            strcpy(token.lexeme, ")");
            i++;
        } else if (isalpha(input_str[i])) {
            token.type = TOKEN_ID;
            int j = 0;
            while (isalnum(input_str[i])) {
                token.lexeme[j++] = input_str[i++];
            }
            token.lexeme[j] = '\0';
        } else {
            printf("Error: Invalid token at position %d\n", i);
            exit(1);
        }
        
        input[token_idx++] = token;
    }
    
    // Add end token
    Token end_token;
    end_token.type = TOKEN_END;
    strcpy(end_token.lexeme, "$");
    input[token_idx++] = end_token;
    
    input_size = token_idx;
}

// Get token type as index for action table
int get_token_index(TokenType type) {
    switch (type) {
        case TOKEN_ID: return 0;
        case TOKEN_PLUS: return 1;
        case TOKEN_STAR: return 2;
        case TOKEN_LPAREN: return 3;
        case TOKEN_RPAREN: return 4;
        case TOKEN_END: return 5;
        default: return -1;
    }
}

// Get non-terminal index for goto table
int get_nonterminal_index(char symbol) {
    switch (symbol) {
        case 'E': return 0;
        case 'T': return 1;
        case 'F': return 2;
        default: return -1;
    }
}

// Print the current stack for display
void print_stack() {
    for (int i = 0; i <= top; i++) {
        printf("%d%c ", stack[i].state, stack[i].symbol);
    }
}

// Print the remaining input for display
void print_input(int current_index) {
    for (int i = current_index; i < input_size; i++) {
        printf("%s", input[i].lexeme);
    }
}

// Parse the input
void parse(char *input_str) {
    // Tokenize input
    tokenize(input_str);
    
    // Initialize stack
    push(0, '$');
    
    printf("\nParsing Steps:\n");
    printf("------------------------------------------------\n");
    printf("%-20s %-20s %-20s\n", "Stack", "Input", "Action");
    printf("------------------------------------------------\n");
    
    int current_index = 0;
    
    while (1) {
        int current_state = stack[top].state;
        TokenType token_type = input[current_index].type;
        int token_index = get_token_index(token_type);
        
        // Print stack
        printf("%-20s ", "Stack:");
        print_stack();
        printf("\n");
        
        // Print input
        printf("%-20s ", "Input:");
        print_input(current_index);
        printf("\n");
        
        char action_type = action_table[current_state][token_index].action;
        int action_value = action_table[current_state][token_index].value;
        
        if (action_type == 's') {
            // Shift action
            printf("%-20s Shift %d\n", "Action:", action_value);
            
            char symbol;
            switch (token_type) {
                case TOKEN_ID: symbol = 'i'; break;
                case TOKEN_PLUS: symbol = '+'; break;
                case TOKEN_STAR: symbol = '*'; break;
                case TOKEN_LPAREN: symbol = '('; break;
                case TOKEN_RPAREN: symbol = ')'; break;
                case TOKEN_END: symbol = '$'; break;
                default: symbol = '?';
            }
            
            push(action_value, symbol);
            current_index++;
            
        } else if (action_type == 'r') {
            // Reduce action
            printf("%-20s Reduce by %c -> %s\n", "Action:", 
                   productions[action_value].lhs, 
                   productions[action_value].rhs);
            
            // Pop |β| symbols
            int rhs_len = productions[action_value].rhs_len;
            
            // Special handling for 'id' which is considered as 1 symbol in the grammar
            if (action_value == 6) {  // F -> id
                rhs_len = 1;
            }
            
            for (int i = 0; i < rhs_len; i++) {
                pop();
            }
            
            // Get top state after popping
            int top_state = stack[top].state;
            
            // Push A
            char lhs = productions[action_value].lhs;
            int nt_index = get_nonterminal_index(lhs);
            int goto_state = goto_table[top_state][nt_index];
            
            printf("%-20s Goto[%d,%c] = %d\n", "Goto:", top_state, lhs, goto_state);
            
            push(goto_state, lhs);
            
        } else if (action_type == 'a') {
            // Accept action
            printf("%-20s Accept\n", "Action:");
            printf("\nInput string accepted!\n");
            break;
            
        } else {
            // Error
            printf("%-20s Error\n", "Action:");
            printf("\nSyntax error at token: %s\n", input[current_index].lexeme);
            break;
        }
        
        printf("------------------------------------------------\n");
    }
}

// Print the action table for reference
void print_action_table() {
    printf("Action Table:\n");
    printf("    | id  | +   | *   | (   | )   | $   |\n");
    printf("----+-----+-----+-----+-----+-----+-----+\n");
    
    for (int i = 0; i < 12; i++) {
        printf("%2d | ", i);
        for (int j = 0; j < 6; j++) {
            if (action_table[i][j].action == 's') {
                printf("s%-3d|", action_table[i][j].value);
            } else if (action_table[i][j].action == 'r') {
                printf("r%-3d|", action_table[i][j].value);
            } else if (action_table[i][j].action == 'a') {
                printf("acc |");
            } else {
                printf("    |");
            }
        }
        printf("\n");
    }
}

// Print the goto table for reference
void print_goto_table() {
    printf("\nGoto Table:\n");
    printf("    | E   | T   | F   |\n");
    printf("----+-----+-----+-----+\n");
    
    for (int i = 0; i < 12; i++) {
        printf("%2d | ", i);
        for (int j = 0; j < 3; j++) {
            if (goto_table[i][j] != 0) {
                printf("%-4d|", goto_table[i][j]);
            } else {
                printf("    |");
            }
        }
        printf("\n");
    }
}

int main() {
    char input_str[100];
    
    printf("Enter an expression (e.g., id+id*id, (id+id)*id): ");
    fgets(input_str, sizeof(input_str), stdin);
    
    // Remove newline character
    input_str[strcspn(input_str, "\n")] = '\0';
    
    printf("\nSLR(1) Parser for Grammar:\n");
    printf("E -> E+T | T\n");
    printf("T -> T*F | F\n");
    printf("F -> (E) | id\n\n");
    
    // Print the action and goto tables
    print_action_table();
    print_goto_table();
    
    // Parse the input
    parse(input_str);
    
    return 0;
}
