#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<ctype.h>
char *keywords[33] = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
    "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while"
};
int is_keyword(char *word) {
    for (int i = 0; i < 33; i++) {
        if (strcmp(word, keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}
int is_num(char* word)
{
	for(int i=0;i<strlen(word);i++)
	{
		if (!(word[i] >= '0' && word[i] <= '9'))
			return 0;
	}
	return 1;
}
int main()
{
	char c,buf[1024];
	char id[1024],sl[1024];
	FILE *fp=fopen("test.c","r");
	if (fp == NULL)
	{
		printf("Cannot open file \n");
		exit(0);
	}
	c = fgetc(fp);
	while(c!=EOF)
	{
		int i=0;
		while(c != '\n' && c != EOF)
		{
			buf[i++] = c;
			c = fgetc(fp);
		}
		buf[i] = '\0';
		int j = 0;
		while(buf[j] != '\0')
		{
			while(isspace(buf[j])) j++;
			int k=0;
			id[k] = '\0';
			while((isalpha(buf[j]) || isdigit(buf[j])) && !isspace(buf[j]))
			{
				id[k++] = buf[j++];
			}
			id[k] = '\0';
			if(is_keyword(id))
			{
				printf("Keyword: %s\n",id);
			}
			else if(id[0] != '\0' && id[0] != '=' && id[0] != '+' && id[0] != '-' && id[0] != '*' && id[0] != '/' && id[0] != '>' && id[0] != '<' && id[0] != '!' && id[0] != '%')
			{
				if(is_num(id))
				{
					printf("Numeric constant: %s\n",id);
				}
				else
				{
					printf("Identifier: %s\n",id);
				}
			}
			if(buf[j] == '(' || buf[j] == ')' || buf[j] == ';')
			{
				
				printf("Special symbol: %c \n",buf[j]);
			}
			if(buf[j] == '"')
			{
				int l = 0;
				sl[l++] = buf[j++];
				while(buf[j] != '"') 
					sl[l++] = buf[j++];
				sl[l++] = '"';
				sl[l] = '\0';
				printf("String literal %s\n",sl);
			}
			j++;
		}
		c = fgetc(fp);
	}
	fclose(fp);
	return 0;
}