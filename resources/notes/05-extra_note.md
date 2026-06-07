@property treat method like attribute (without using getter and setter)  

@lru_cached   
 "LRU" stands for Least Recently Used - it automatically discards the least used items when the cache gets full.   

Notice fibonacci(2) was only calculated once, even though it was needed multiple times.     


Field() add constraints when instantiate an object to an attribute     
Field() also used for metadata which allow BaseModel don't validate those fields as they only give description about the object attributes   


Security 
Reject malicious input which intent to reveal secret information   
Take note of sensitive input like credit card info etc.  
Output validation, check if the output if appropriate before sending it to user's   

Annotated = creating a reusable "type with rules" that you can use everywhere, instead of typing Field(...) over and over.  


Caching to reduce api cost  
Monitoring layer for future improvements   

Security checklist  
Reliability checklist  
Performance checklist   
Observability checklist  
Deployment checklist  