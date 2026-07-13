from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name : str = "pratik" # this works as defalut
    age : Optional[int] = None # this will print None as defalut
    email : EmailStr 
    cgpa : float = Field(gt=0,lt=10,default=5,description='A decimal value representing cgpa of the student')


new_student = {'age':'19','email':'pratik531@gmail.com'} # or name you can add inside the curlybracket like --> new_student = {'name':'pratik'}
student = Student(**new_student)    # output will be same in both cases

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()

print(student.email)