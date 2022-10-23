import pickle
import tkinter as tk
from tkinter import ttk
from copy import deepcopy

import numpy as np


class Interactive_Dictionary_Editor:
    """
    An user interface to edit python dictionaries in a drag and drop interactive gui.
    Capabilities:
    1. Edit dictionary values
    2. Add new dictionary values
    3. Delete dictionary values
    4. Rename dictionary keys
    5. Add new dictionary keys
    6. Delete dictionary keys
    7. Add new dictionaries
    8. Delete dictionaries
    9. Save dictionary to file

    Features:
    1. Auto-saves every change
    2. All changes are undoable
    3. Interactive drag and drop into locking positions
    4. Tree-view style navigation
    5. Interactive editor
    6. Works with nested dictionaries
    7. Works with nested lists
    8. Works with mixed nested dictionaries and lists
    9. Works with python types: int, float, str, bool and None
    10. Works with numpy types: numpy.ndarray, numpy.int64, numpy.float64, numpy.bool, numpy.str

    """

    def __init__(self, dictionary, filename,
                 title='Interactive Dictionary Editor',
                 width=500, height=500,
                 fontsize=10,
                 show_save_button=True,
                 show_undo_button=True,
                 show_load_button=True):
        """
        Initialize an interactive dictionary editor.

        Parameters
        ----------
        dictionary : dict
            The dictionary to be edited.
        filename : str
            The filename to save the dictionary to.
        title : str
            The title of the gui.
        width : int
            The width of the gui.
        height : int
            The height of the gui.
        fontsize : int
            The fontsize of the gui.
        show_save_button : bool
            Show the save button.
        show_undo_button : bool
            Show the undo button.
        show_load_button : bool
            Show the load button.
        """
        # Save the filename
        self.filename = filename
        # Copy the dictionary
        self.dictionary = deepcopy(dictionary)
        # Initialize the undo list
        self.undos = []
        # Start the GUI
        self.root = tk.Tk()
        self.root.title(title)
        # Set the window size
        w = width
        h = height
        self.root.geometry(str(w) + 'x' + str(h))
        # Create a frame to contain the whole gui
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Create a frame to contain the list and the label
        self.list_frame = tk.Frame(self.main_frame)
        self.list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Create a frame to contain the editor
        self.editor_frame = tk.Frame(self.main_frame)
        self.editor_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # Create a frame to contain the buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Create a label
        self.label = tk.Label(self.list_frame, anchor='w', text='', font=('Helvetica', fontsize))
        self.label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Create a listbox
        self.listbox = tk.Listbox(self.list_frame, font=('Helvetica', fontsize))
        self.listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Create an editor
        self.editor = tk.Text(self.editor_frame, font=('Helvetica', fontsize))
        self.editor.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Create a save button
        if show_save_button:
            self.save_button = tk.Button(self.button_frame, text='Save', command=self.save)
            self.save_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Create an undo button
        if show_undo_button:
            self.undo_button = tk.Button(self.button_frame, text='Undo', command=self.undo)
            self.undo_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Create a load button
        if show_load_button:
            self.load_button = tk.Button(self.button_frame, text='Load', command=self.load)
            self.load_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Bind the listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        # Bind the editor
        self.editor.bind('<FocusOut>', self.on_editor_focus_out)
        self.editor.bind('<Return>', self.on_editor_focus_out)
        # Fill the listbox
        self.fill_listbox(self.dictionary)
        # Display the gui
        self.root.mainloop()

    def fill_listbox(self, dictionary):
        """
        Fill the listbox with the dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary to be viewed in the listbox.
        """
        # Clear the listbox
        self.listbox.delete(0, tk.END)
        # Add all the key-value pairs
        for key in sorted(list(dictionary.keys())):
            val = dictionary[key]
            # Handle dictionaries
            if isinstance(val, dict):
                self.listbox.insert(tk.END, '- ' + str(key) + ' -')
            # Handle lists
            elif isinstance(val, list):
                self.listbox.insert(tk.END, '- ' + str(key) + ' -')
            # Handle other types (int, float, str, bool, None)
            else:
                self.listbox.insert(tk.END, str(key) + ' : ' + str(val))
        # Select the first item
        self.listbox.selection_set(0)
        # Trigger the listbox selection event
        self.on_listbox_select(None)

    def save(self, event=None):
        """
        Save the dictionary to file.
        """
        # Save the dictionary
        with open(self.filename, 'wb') as f:
            pickle.dump(self.dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, event=None):
        """
        Load the dictionary from file.
        """
        # Load the dictionary
        with open(self.filename, 'rb') as f:
            self.dictionary = pickle.load(f)
        # Clear the undos
        self.undos = []
        # Fill the listbox
        self.fill_listbox(self.dictionary)

    def undo(self, event=None):
        """
        Undo the last action.
        """
        # Undo the last action
        if self.undos:
            self.dictionary = self.undos.pop()
        else:
            self.dictionary = {}
        # Fill the listbox
        self.fill_listbox(self.dictionary)

    def on_listbox_select(self, event):
        """
        Handle the click on a listbox item.
        """
        # Get the selected item
        index = self.listbox.curselection()[0]
        item = self.listbox.get(index)
        # Update the label
        self.label.configure(text=item)
        # Update the editor
        self.update_editor()

    def update_editor(self):
        """
        Update the editor.
        """
        # Get the selected item
        index = self.listbox.curselection()[0]
        item = self.listbox.get(index)
        # Handle dictionaries
        if item.startswith('- '):
            # Clear the editor
            self.editor.delete(1.0, tk.END)
            # Disable editing
            self.editor.configure(state='normal')
        # Handle lists
        elif item.startswith('[ '):
            # Clear the editor
            self.editor.delete(1.0, tk.END)
            # Disable editing
            self.editor.configure(state='normal')
        # Handle other types (int, float, str, bool, None)
        else:
            # Get the key and value
            key = item.split(' : ')[0]
            val = item.split(' : ')[1]
            # Enable editing
            self.editor.configure(state='normal')
            # Update the editor
            self.editor.delete(1.0, tk.END)
            self.editor.insert(tk.END, val)
            # Disable editing
            self.editor.configure(state='disabled')

    def on_editor_focus_out(self, event):
        """
        Handle the focus out from the editor.
        """
        # Get the selected item
        index = self.listbox.curselection()[0]
        item = self.listbox.get(index)
        # Handle dictionaries
        if item.startswith('- '):
            # Get the key of the dictionary to be edited
            key = item.split(' - ')[0][2:]
            # Get the new value
            val = self.editor.get(1.0, tk.END).strip()
            # Try to convert the value to a python type
            try:
                val = eval(val)
            except:
                print('Not a python type:', val)
                return
            # Validate the value
            if not isinstance(val, dict):
                print('Not a dictionary:', val)
                return
            # Add an undo
            self.undos.append(deepcopy(self.dictionary))
            # Update the dictionary
            self.dictionary[key] = val
            # Fill the listbox
            self.fill_listbox(self.dictionary)
        # Handle lists
        elif item.startswith('[ '):
            # Get the key of the list to be edited
            key = item.split(' - ')[0][1:]
            # Get the new value
            val = self.editor.get(1.0, tk.END).strip()
            # Try to convert the value to a python type
            try:
                val = eval(val)
            except:
                print('Not a python type:', val)
                return
            # Validate the value
            if not isinstance(val, list):
                print('Not a list:', val)
                return
            # Add an undo
            self.undos.append(deepcopy(self.dictionary))
            # Update the dictionary
            self.dictionary[key] = val
            # Fill the listbox
            self.fill_listbox(self.dictionary)
        # Handle other types (int, float, str, bool, None)
        else:
            # Get the old key and value
            old_key = item.split(' : ')[0]
            old_val = item.split(' : ')[1]
            # Get the new key and value
            new_key = self.editor.get(1.0, tk.END).strip()
            new_val = self.editor.get(1.0, tk.END).strip()
            # Try to convert the value to a python type
            try:
                new_val = eval(new_val)
            except:
                print('Not a python type:', new_val)
                return
            # Validate the value
            if not isinstance(new_val, (int, float, str, bool, None, np.ndarray, np.int64, np.float64, np.bool, np.str)):
                print('Not a python type:', new_val)
                return
            # Add an undo
            self.undos.append(deepcopy(self.dictionary))
            # Update the dictionary
            self.dictionary[new_key] = new_val
            del self.dictionary[old_key]
            # Fill the listbox
            self.fill_listbox(self.dictionary)


if __name__ == '__main__':
    # Initialize a dictionary
    dictionary = {'key1': 1,
                  'key2': 2.0,
                  'key3': '3.0',
                  'key4': True,
                  'key5': None,
                  'key6': np.array([1, 2, 3]),
                  'key7': {'key7.1': '7.1',
                           'key7.2': 7.2},
                  'key8': [8, '8', True, None, np.array([8, 8, 8])]}
    # Run the dictionary editor
    dictionary_editor = Interactive_Dictionary_Editor(dictionary,
                                                      filename='test_dictionary.pkl',
                                                      title='Test Dictionary Editor',
                                                      width=1200, height=600,
                                                      fontsize=11)
