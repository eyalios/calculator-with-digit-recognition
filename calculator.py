from tkinter import filedialog
import pickle
import learner
import cnn
import re
import  tkinter as tk
from PIL import Image
import numpy as np
import scipy.misc
import tkinter.messagebox as msgbox
import matplotlib.pyplot as plt
class Calc():


    def __init__(self):
        self.win = tk.Tk()
        self.button_dict = {}
        self.actions_list = ['*', '/', '+', '-', '.', '=']
        self.EQUAL_SIGN = '='
        self.DOT_SIGN = '.'
        self.recent_action = "n"
        self.mySVM = learner.Learner()
        self.my_cnn = cnn.cnn()
    def create_gui(self):
        """
            function creates the layout for calculetor gui.
        :return:
        """
        entry = tk.Entry(self.win,state="readonly",readonlybackground="lightblue",background="lightblue")
        entry.grid(sticky="nsew",columnspan=4,row=0)
        #create zero button,=,. buttons
        self.button_dict['0'] = tk.Button(self.win, text=0, width=10,background = 'blue')
        self.button_dict['.'] = tk.Button(self.win, text='.', width=10,background = 'blue')
        self.button_dict['='] = tk.Button(self.win, text='=', width=10,background = 'red')
        self.button_dict['0'].grid(row=4, column=1,sticky="nsew")
        self.button_dict['.'].grid(row=4, column=0,sticky="nsew")
        self.button_dict['='].grid(row=4, column=2,sticky="nsew")
        self.button_dict['0'].bind('<Button-1>', self.button_clicked)
        self.button_dict['.'].bind('<Button-1>', self.button_clicked)
        self.button_dict['='].bind('<Button-1>', self.button_clicked)
        #canvas button creation
        self.button_dict["canvas"] = tk.Button(self.win, text='canvas', width=10, background='blue')
        self.button_dict["canvas"].grid(row=5, column=1, sticky="nsew")
        self.button_dict["canvas"].bind('<Button-1>', self.create_canvas_window)
        #buttons 1-9
        for index in range(0,9):
            self.button_dict[str(index+1)] = tk.Button(self.win,text = (index+1),width = 10,background = 'blue')
            self.button_dict[str(index+1)].bind('<Button-1>', self.button_clicked)
            self.button_dict[str(index+1)].grid(row = int(index/3) + 1, column = index%3,sticky="nsew")
        index = 0
        #action buttons creation.
        for action in calc.actions_list:
            if(action != self.EQUAL_SIGN and action != self.DOT_SIGN):
                self.button_dict[action] = tk.Button(self.win,text= action,width=10,background = 'red')
                self.button_dict[action].bind('<Button-1>', self.button_clicked)
                self.button_dict[action].grid(row=index + 1 ,column = 3,sticky="nsew")
                index += 1
                self.win.grid_columnconfigure(list(range(10)), weight=2)
                self.win.grid_rowconfigure(list(range(10)), weight=2)

        self.win.bind('<KeyPress>', self.flash)
        self.win.mainloop()

    def button_clicked(self,event):
        """
        activates the action and digit buttons
        :param event:
        :return:
        """
        #extract the text of the button clicked.
        button_text = str(event.widget.cget('text'))
        #add to entry widget on calculator
        entry_widget = self.win.winfo_children().__getitem__(0)
        #compute if '='
        if(button_text == "="):
            self.compute(entry_widget)
        else:
            self.add_to_entry_widget(button_text,entry_widget)


    def add_to_entry_widget(self,text,entry_widget):

      """
      @:param text- the text to be added to the entry widget
      @:param entery_widget- the given entry widget
      function adds the char to the calcultor top, if it is ok with syntax rules.
      """
      widget_text = entry_widget.get()
      #check that no two non numerical chars come in sequance
      if (widget_text and text in self.actions_list and widget_text[-1]  in self.actions_list):
          return
      #check that a non numerical char is not first.
      if(not widget_text and text in self.actions_list):
          return
      if(text == self.DOT_SIGN and self.recent_action == self.DOT_SIGN):
          return
      if(text in self.actions_list):
          self.recent_action = text
      #add the char to entry
      entry_widget.configure(state='normal')
      entry_widget.insert('end', text)
      entry_widget.configure(state='readonly')

    def set_entry_widget(self,str,entry_widget):
        """

        :param str: the string to set the entry widget to.
        :param entry_widget: given entry widget of calculator
        :return:
        """
        #enable access to entry widget
        entry_widget.configure(state='normal')
        #delete all widget and write in str.
        entry_widget.delete(0,'end')
        entry_widget.insert(0,str)
        #disable access to entry widget.
        entry_widget.configure(state='readonly')

    def flash(self,event):
        """
          simulates a button click when a keyboard is struck.
          """
        #getting with key was struck.
        key_struck = event.char
        entry_widget = self.win.winfo_children().__getitem__(0)
        #checking which color the key is to flash it correctly and activate relevnt events.
        #for all buttons switching color for milisecond and switching back to orignial color.
        if(key_struck == "="):
            self.compute(entry_widget)
            self.win.after(100, lambda: self.button_dict[key_struck].config(bg='red'))
            return
        if(key_struck in self.button_dict):
            self.button_dict[key_struck].config(bg = 'lightgrey')
            if(key_struck in self.actions_list):
                self.add_to_entry_widget(key_struck,entry_widget)
                self.win.after(100, lambda: self.button_dict[key_struck].config(bg='red'))
            else:
                self.add_to_entry_widget(key_struck, entry_widget)
                self.win.after(100, lambda: self.button_dict[key_struck].config(bg='blue'))

    def open_image_file(self,event):
        """
        opens a chosen image file. and classifies it with my svm implemination
        and then offers the user to add the digit to entry widget or try again.
        :param event:
        :return: img - the image opened as flatten binary np vector.
        """
        self.draw_pad.update()
        #open file from dialog box, convert it to np binary vector.
        fileName = filedialog.askopenfilename()
        img = Image.open(fileName).convert('1')
        img = np.array(img).reshape(28*28)
        #save loaded image as np array.
        np.save(re.split('[.]',fileName)[0] + '.npy', img)
        #classify image using mySVM implemintaion
        res = self.mySVM.classify(re.split('[.]',fileName)[0]+'.npy')
        #output res to user and check if good to add to entry widget.
        self.check_digit_with_user(res)
        return img

    def check_digit_with_user(self,digit):
        """
        output message box to user with given digit, if "yes" is clicked, will add to widget entry
        if "no" will do nothing.
        :param digit:
        :return:
        """
        msgBox = msgbox.askquestion('res', 'was ' + str(digit) + ' the digit? if not please try again')
        if msgBox == 'yes':
            self.add_to_entry_widget(str(digit), self.win.winfo_children().__getitem__(0))
            self.draw_pad.destroy()
        else:
            print("no")
        return

    def compute(self,entry_widget):
        """
        computes and presents the arithmetic action
        :param entry_widget:
        :return:
        """
        text = entry_widget.get()
        result = ""
        #parse input to digit list
        parsed_text = re.split(r'[*/+-]',text)
        #parse input to action list.
        parsed_actions = re.split('[0-9\\.]+',text)
        num_actions = len(parsed_actions)-1
        parsed_actions = parsed_actions[1:num_actions]
        #first loaction will be a list of  the * and / actions
        first_location = []
        #second location will be a list of the - and + actions.
        second_location =[]
        counter = 0
        #adding actions to relevnt lists.
        for action in parsed_actions:
            if(action  == '*' or action == '/'):
                first_location += [counter]
            else:
                second_location += [counter]
            counter += 1
        counter = 0
        #first compute the first in order (first location in list of action - *,/)
        for location in first_location:
            if(parsed_actions[location] == '*'):
                result = float(parsed_text[location-counter]) * float(parsed_text[location + 1 - counter])
            else:
                result = float(parsed_text[location-counter]) / float(parsed_text[location + 1 - counter])
            #cheking if this is the last action, updating the number list to be the given resualt.
            if(location == (num_actions)):
                parsed_text = parsed_text[:location-counter] + [result]
            #if not, updating the number to be the result and the rest of the numbers to compute.
            else:
                parsed_text = parsed_text[:location-counter] + [result] + parsed_text[location+2-counter:]
        #now we have a list of all unified multipls and divisions
            counter += 1
        #now add and subtract
        counter = 0
        for location in second_location:
            if(parsed_actions[location]=='+'):
                result = float(parsed_text[counter]) + float(parsed_text[counter+1])
            else:
                result = float(parsed_text[counter]) -  float(parsed_text[counter + 1])
            parsed_text =  [result] + parsed_text[2:]
        #if there is only number and no actions to compute.
        if(not result and parsed_text[0]):
            result =  parsed_text[0]
        #update the entry widget
        self.set_entry_widget(result,entry_widget)

    def getter(self,event):
        """
        saves the image drawn on to the canvas.
        :param event:
        :return:
        """
        self.save_image_to_classify('try')

    def paint(self,event):
        """
        draws a point on the canvas
        :param event:
        :return:
        """
        #the radius of the oval to draw.
        r = 1
        x1, y1 = ( event.x - r ), ( event.y - r )
        x2, y2 = ( event.x + r ), ( event.y + r )
        self.canvas.create_oval( x1, y1, x2, y2, fill = "black")


    def create_canvas_window(self,event):
      """
      create and open pop up canvas window.
      :param event:
      :return:
      """
      #create elemnts of popwindow, including canvas, and buttons for diffrent classifiers.
      self.draw_pad = tk.Tk()
      self.canvas = tk.Canvas(self.draw_pad, width=26, height=26, bg="white")
      self.draw_pad.geometry("200x400")
      clear_button = tk.Button(self.draw_pad, text='clear', width=10, background='blue')
      save_button = tk.Button(self.draw_pad, text='save', width=10, background='red')
      notMySVM_button = tk.Button(self.draw_pad, text='notmySVM', width=10, background='yellow')
      mySVM_button = tk.Button(self.draw_pad, text='MySVM', width=10, background='green')
      openIm_button = tk.Button(self.draw_pad, text='open image- mySVM', width=10, background='pink')
      cnn_button = tk.Button(self.draw_pad, text='cnn with keras', width=10, background='cyan')
      #binding functions to each button
      self.canvas.bind("<B1-Motion>", self.paint)
      clear_button.bind('<Button-1>', self.clear_canvas)
      save_button.bind('<Button-1>', self.getter)
      mySVM_button.bind('<Button-1>', self.identify_mySVM)
      notMySVM_button.bind('<Button-1>', self.identify_NotMySVM)
      openIm_button.bind('<Button-1>', self.open_image_file)
      cnn_button.bind('<Button-1>', self.identify_cnn)

     #placing buttons on drawpad window.
      clear_button.grid(padx=15, pady=2, sticky="nsew")
      save_button.grid(padx=15, pady=2, sticky="nsew")
      mySVM_button.grid(padx=15, pady=2, sticky="nsew")
      notMySVM_button.grid(padx=15, pady=2, sticky="nsew")
      openIm_button.grid(padx=15, pady=2, sticky="nsew")
      cnn_button.grid(padx=15, pady=2, sticky="nsew")
      self.canvas.grid( padx=75, pady=75,sticky="nsew")
      self.draw_pad.mainloop()


    def save_image_to_classify(self,fileName):
        """
        save image from canvas in revese (from original) binary np array.
        :param fileName:
        :return:
        """
        widget = self.canvas
        widget.postscript(file="C:/Users/USER/Downloads/mnistasjpg/try.eps")
        # open saved canvas as eps
        img = Image.open("C:/Users/USER/Downloads/mnistasjpg/try.eps")
        # save canvas as jpeg to convert it.
        img.save(fileName + '.jpg', "JPEG")
        # open image as jpg.
        img = Image.open(fileName + '.jpg').convert('1')
        img = np.array(img)
        # make oppsite image to canvas to match format of database.
        img = np.absolute(1 - img)
        #zero padding image to fit 28*28 format
        img = np.pad(img, 2, mode='constant')
        # save reversd image.
        np.save(fileName + '.npy', img)
        return img

    def identify_cnn(self,event):
        """
        using my cnn cllasifier to classify image from canvas. also saving image
        in as binary and reverse np array.
        :param event:
        :return:
        """
        self.save_image_to_classify('my_cnn')
        # classify
        res = self.my_cnn.classify('C:/Users/USER/playground/my_cnn.npy')
        # check with user.
        self.check_digit_with_user(res)

    def identify_mySVM(self,event):
        """
        using my SVM classifier to classify image from canvas. also saving image
        in as binary and reverse np array.
        :param event:
        :return:
        """
        #save the image.
        self.save_image_to_classify('my_svm')
        #classify
        res = self.mySVM.classify('C:/Users/USER/playground/my_svm.npy')
        #check with user.
        self.check_digit_with_user(res)

    def identify_NotMySVM(self,event):
        """
        using the sklearn learn svm to classify digit.
        :param event:
        :return:
        """
        #save image from canvas
        img = self.save_image_to_classify('notMy_svm')
        #get the model learned before using sklearn.
        model = pickle.load(open('finalized_model.sav', 'rb'))
        #use model to predict image.
        pred1 = model.predict(img.reshape(-1,1).T)
        #check with user.
        self.check_digit_with_user(str(pred1[0]))


    def clear_canvas(self,event):
        """
        clear canvas from all objects.
        :param event:
        :return:
        """
        self.canvas.delete("all")


calc = Calc()
calc.create_gui()
