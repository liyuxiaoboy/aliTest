struct ListNode {
      int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
 };
 
class Solution { 
    public: ListNode* reverseKGroup(ListNode* head, int k) { 
        ListNode* pre=head; 
        int count=0; 
        while(pre!=NULL&&count<k){ 
            pre=pre->next; 
            count++;
        } 
        if(count==k) { 
            pre=reverseKGroup(pre,k); 
            while(count>0){
       
                ListNode* temp=head->next;
                head->next=pre;
                pre=head;
                head=temp;
                count--; 
            } 
            head=pre; 
        } 
        return head; 
    } 
};
