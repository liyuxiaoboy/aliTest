#include <iostream>
#include <string>
using namespace std;

class Solution{
public:
    bool isPalindrome(string s){
        if(s.empty()) {
			return true;
		}
        int i=0;
		int j=s.length()-1;
        while(i<j){
            if(notlegalchar(s[i])){
				i++;
				continue;
			}
            if(notlegalchar(s[j])){
				j--;
				continue;
			} 
            if(CheckWord(s[i]) != CheckWord(s[j])){
				return false;
			}
			else{
				i++;
				j--;
			}
        }
        return true;
    }

private:
    bool notlegalchar(char a){
        if((a>='a'&& a<='z')||(a>='A'&&a<='Z')||(a>='0'&&a<='9')) {
			return false;
		}
		return true;
		}
	char CheckWord(char word){
		if(word>='A'&&word<='Z'){
			return word-'A'+'a';
		}else{
			return word;
		}

	}
};