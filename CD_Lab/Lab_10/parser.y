%{
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void yyerror(const char *s);
int yylex(void);

// Stack for postfix evaluation
#define MAX_STACK 100
int stack[MAX_STACK];
int top = -1;

void push(int val) {
    if (top < MAX_STACK - 1) {
        stack[++top] = val;
    } else {
        yyerror("Stack overflow");
    }
}

int pop() {
    if (top >= 0) {
        return stack[top--];
    } else {
        yyerror("Stack underflow");
        return 0;
    }
}
%}

%union {
    int ival;
    char *sval;
}

%token <ival> NUMBER
%token <sval> ID
%token PLUS MINUS MULT DIV POWER NEWLINE
%token DECLARE IF THEN ELSE ENDIF

%type <ival> exp decision condition

%%

input: 
    | input line
;

line: 
    '\n'
    | declaration '\n' { printf("Valid Declaration\n"); }
    | exp '\n' { 
        printf("Valid Expression. Result = %d\n", $1); 
        top = -1; // Reset stack 
    }
    | decision '\n' { 
        printf("Valid Decision Making Statement. Result = %d\n", $1); 
    }
;

declaration: 
    DECLARE ID { 
        printf("Declared variable: %s\n", $2); 
        free($2); 
    }
;

decision: 
    IF condition THEN exp ELSE exp ENDIF {
        // If condition is non-zero, take first exp, else take second
        $$ = $2 ? $4 : $6;
    }
;

condition:
    exp exp MULT {
        int b = pop(), a = pop();
        int result = a * b;
        push(result);
        $$ = result;
        printf("Condition Multiplication: %d * %d = %d\n", a, b, result);
    }
    | exp exp DIV {
        int b = pop(), a = pop();
        if (b != 0) {
            int result = a / b;
            push(result);
            $$ = result;
            printf("Condition Division: %d / %d = %d\n", a, b, result);
        } else {
            yyerror("Division by zero");
            $$ = 0;
        }
    }
;

exp: 
    NUMBER { 
        push($1); 
        $$ = $1; 
    }
    | exp exp PLUS { 
        int b = pop(), a = pop(); 
        int result = a + b; 
        push(result); 
        $$ = result;
        printf("Addition: %d + %d = %d\n", a, b, result);
    }
    | exp exp MINUS { 
        int b = pop(), a = pop(); 
        int result = a - b; 
        push(result); 
        $$ = result;
        printf("Subtraction: %d - %d = %d\n", a, b, result);
    }
    | exp exp MULT { 
        int b = pop(), a = pop(); 
        int result = a * b; 
        push(result); 
        $$ = result;
        printf("Multiplication: %d * %d = %d\n", a, b, result);
    }
    | exp exp DIV { 
        int b = pop(), a = pop(); 
        if (b != 0) {
            int result = a / b; 
            push(result); 
            $$ = result;
            printf("Division: %d / %d = %d\n", a, b, result);
        } else {
            yyerror("Division by zero");
            $$ = 0;
        }
    }
    | exp exp POWER { 
        int b = pop(), a = pop(); 
        int result = pow(a, b); 
        push(result); 
        $$ = result;
        printf("Exponentiation: %d ^ %d = %d\n", a, b, result);
    }
    | exp 'n' { 
        int a = pop(); 
        int result = -a; 
        push(result); 
        $$ = result;
        printf("Negation: -%d = %d\n", a, result);
    }
;

%%

int main() {
    printf("Enter expressions (Ctrl+D to stop):\n");
    yyparse();
    return 0;
}

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}
