#!/usr/bin/env python

# ---------------------------------------------------------
# Class Viewer
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Written by Erik Bochinski
# --------------- ------------------------------------------

try:
    import gi
except:
    import sys
    sys.path.append("/usr/local/lib/python3.6/site-packages")
    import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf

# import pyGtk
# pyGtk.require('3.0')
# import Gtk

# import Gtk
import sys

from DispGrid import DispGrid


class ClassViewer:
    def close_application(self, *args):
        Gtk.main_quit()

    def __init__(self):
        self.window = Gtk.Window()
        self.window.connect("delete_event", self.close_application)
        self.window.set_default_size(900, 640)
        #self.window.set_border_width(6)
        main_vbox = Gtk.VBox()
        self.window.add(main_vbox)

        # create the top menu
        file_menu = Gtk.Menu()

        # Create the menu items
        open_item = Gtk.MenuItem("Open")
        save_item = Gtk.MenuItem("Save")
        saveas_item = Gtk.MenuItem("Save as..")
        export_item = Gtk.MenuItem("Export")
        quit_item = Gtk.MenuItem("Quit")

        # Add them to the menu
        file_menu.append(open_item)
        file_menu.append(save_item)
        file_menu.append(saveas_item)
        file_menu.append(export_item)
        file_menu.append(quit_item)

        # Attach the callback functions to the activate signal
        open_item.connect_object("activate", self.menuitem_response, "file.open")
        export_item.connect_object("activate", self.menuitem_response, "file.export")
        saveas_item.connect_object("activate", self.menuitem_response, "file.save")
        save_item.connect_object("activate", self.menuitem_save_response, "file.save")

        # We can attach the Quit menu item to our exit function
        quit_item.connect_object("activate", Gtk.main_quit, "file.quit")

        menu_bar = Gtk.MenuBar()
        main_vbox.pack_start(menu_bar, expand=False, fill=False, padding=0)

        file_item = Gtk.MenuItem("File")
        file_item.show()
        file_item.set_submenu(file_menu)

        menu_bar.append(file_item)

        help_menu = Gtk.Menu()
        about_item = Gtk.MenuItem("About")
        about_item.connect("activate", self.show_about)
        help_menu.append(about_item)
        help_item = Gtk.MenuItem("Help")
        help_item.set_submenu(help_menu)
        menu_bar.append(help_item)

        # grid for base layout
        self.main_grid = Gtk.Grid()
        #self.main_grid.set_row_spacing(6)
        self.main_grid.set_column_spacing(6)
        main_vbox.pack_start(self.main_grid, expand=True, fill=True, padding=0)
        # self.window.add(self.main_grid)


        # create vbox for right part
        vbox_right = Gtk.VBox(spacing=6)
        self.main_grid.attach(vbox_right, 1, 0, 1, 1)

        # add spinbuttons
        spinbuttons_frame = Gtk.Frame(label="Layout")
        spinbuttons_grid = Gtk.Grid()
        spinbuttons_frame.add(spinbuttons_grid)
        #vbox_right.add(spinbuttons_frame)
        vbox_right.pack_start(spinbuttons_frame, expand=False, fill=False, padding=0)

        spinbuttons_grid.attach(Gtk.Label("Rows:"), 0, 0, 1, 1)
        spinbuttons_grid.attach(Gtk.Label("Cols:"), 0, 1, 1, 1)

        adjustment_rows = Gtk.Adjustment(value=3, lower=1, upper=10, step_incr=1)
        adjustment_cols = Gtk.Adjustment(value=3, lower=1, upper=10, step_incr=1)
        self.spinbutton_rows = Gtk.SpinButton(adjustment=adjustment_rows)
        self.spinbutton_cols = Gtk.SpinButton(adjustment=adjustment_cols)
        self.spinbutton_rows.set_value(3)
        self.spinbutton_cols.set_value(3)
        self.spinbutton_rows.connect("value_changed", self.update_grid)
        self.spinbutton_cols.connect("value_changed", self.update_grid)
        spinbuttons_grid.attach(self.spinbutton_rows, 1, 0, 1, 1)
        spinbuttons_grid.attach(self.spinbutton_cols, 1, 1, 1, 1)

        # add class selection
        class_selection_frame = Gtk.Frame(label="Class Selection")
        vbox_right.pack_start(class_selection_frame, expand=False, fill=False, padding=0)
        vbox_class_selection = Gtk.VBox()
        class_selection_frame.add(vbox_class_selection)

        class_selection_store = Gtk.ListStore(str)
        class_selection_store.append(["All"])
        # class_selection_store.append(["NUE"])
        # class_selection_store.append(["TUB"])
        self.class_combo = Gtk.ComboBox.new_with_model_and_entry(class_selection_store)
        self.class_combo.set_entry_text_column(0)
        self.class_combo.set_active(0)
        self.class_combo.connect("changed", self.update_grid)
        vbox_class_selection.pack_start(self.class_combo, expand=False, fill=False, padding=0)

        self.radiobutton_unlabelled = Gtk.RadioButton('unlabelled')
        self.radiobutton_unlabelled.connect("toggled", self.radio_toggle, 'unlabelled')
        vbox_class_selection.pack_start(self.radiobutton_unlabelled, False, False, 0)

        radiobutton_labelled = Gtk.RadioButton('labelled', group=self.radiobutton_unlabelled)
        vbox_class_selection.pack_start(radiobutton_labelled, False, False, 0)

        # add sorting
        sorting_frame = Gtk.Frame(label="Sorting")
        vbox_right.pack_start(sorting_frame, expand=False, fill=False, padding=0)
        vbox_sorting = Gtk.VBox()
        sorting_frame.add(vbox_sorting)


        # TODO get possible sortings from results file
        sorting_store = Gtk.ListStore(str)
        sorting_store.append(["MS"])
        sorting_store.append(["LC"])
        sorting_store.append(["EN"])
        sorting_store.append(["Rand"])
        self.sorting_combo = Gtk.ComboBox.new_with_model_and_entry(sorting_store)
        self.sorting_combo.set_entry_text_column(0)
        self.sorting_combo.set_active(0)
        self.sorting_combo.connect("changed", self.update_grid)
        vbox_sorting.pack_start(self.sorting_combo, False, False, 0)

        self.radiobutton_confidence = Gtk.RadioButton('most confident')
        self.radiobutton_confidence.connect("toggled", self.conf_toggle, 'conf')
        vbox_sorting.pack_start(self.radiobutton_confidence, False, False, 0)

        radiobutton_lconfidence = Gtk.RadioButton('least confident', group=self.radiobutton_confidence)
        vbox_sorting.pack_start(radiobutton_lconfidence, False, False, 0)

        # add statistics
        statistics_frame = Gtk.Frame(label="Statistics")
        vbox_right.pack_start(statistics_frame, expand=False, fill=False, padding=0)
        self.vbox_statistics = Gtk.VBox()
        statistics_frame.add(self.vbox_statistics)

        # label = Gtk.Label("test")
        # self.vbox_statistics.pack_start(label, False, False, 0)

        # add display grid
        self.disp_grid = DispGrid(self.main_grid,
                                  self.get_rows(),
                                  self.get_cols(),
                                  self.get_class(),
                                  self.get_sorting(),
                                  self.get_labelled(),
                                  self.get_confidence())

        # add buttons
        bottom_grid = Gtk.Grid()

        button_prev = Gtk.Button(label="Prev")
        button_prev.connect("clicked", self.on_button_clicked, "prev")
        bottom_grid.attach(button_prev, 0, 0, 1, 1)

        button_next = Gtk.Button(label="Next")
        button_next.connect("clicked", self.on_button_clicked, "next")
        bottom_grid.attach(button_next, 1, 0, 1, 1)

        button_accept = Gtk.Button(label="Accept")
        button_accept.connect("clicked", self.disp_grid.accept)
        bottom_grid.attach(button_accept, 2, 0, 1, 1)

        offset_hbox = Gtk.HBox()
        label = Gtk.Label("Offset: ")
        offset_hbox.pack_start(label, expand=False, fill=False, padding=0)
        self.offset_label = Gtk.Label("0")
        offset_hbox.pack_start(self.offset_label, expand=False, fill=False, padding=0)
        bottom_grid.attach(offset_hbox, 3, 0, 1, 1)

        self.main_grid.attach(bottom_grid, 0, 1, 1, 1)

        # accelerators
        agr = Gtk.AccelGroup()
        self.window.add_accel_group(agr)
        key, mod = Gtk.accelerator_parse("<Control>Q")
        quit_item.add_accelerator("activate", agr, key,
                                  mod, Gtk.AccelFlags.VISIBLE)

        key, mod = Gtk.accelerator_parse("<Control>S")
        save_item.add_accelerator("activate", agr, key,
                                  mod, Gtk.AccelFlags.VISIBLE)

        key, mod = Gtk.accelerator_parse("S")
        button_next.add_accelerator("activate", agr, key,
                                  mod, Gtk.AccelFlags.VISIBLE)

        key, mod = Gtk.accelerator_parse("A")
        button_accept.add_accelerator("activate", agr, key,
                                    mod, Gtk.AccelFlags.VISIBLE)

        # init stuff
        self.update_grid()
        self.window.show_all()

    def radio_toggle(self, widget, data=None):
        self.update_grid()

    def conf_toggle(self, widget, data=None):
        self.update_grid()

    def get_class(self):
        class_iter = self.class_combo.get_active_iter()
        assert class_iter is not None
        class_model = self.class_combo.get_model()
        class_ = class_model[class_iter][0]

        return class_

    def get_sorting(self):
        sorting_iter = self.sorting_combo.get_active_iter()
        assert sorting_iter is not None
        sorting_model = self.sorting_combo.get_model()
        sorting = sorting_model[sorting_iter][0]

        return sorting

    def get_rows(self):
        return int(self.spinbutton_rows.get_value())

    def get_cols(self):
        return int(self.spinbutton_cols.get_value())

    def get_labelled(self):
        return not self.radiobutton_unlabelled.get_active()

    def get_confidence(self):
        return self.radiobutton_confidence.get_active()

    def update_grid(self, reset_offset=True):
        cols = self.get_cols()
        rows = self.get_rows()

        class_ = self.get_class()
        sorting = self.get_sorting()

        labelled = self.get_labelled()
        confidence = self.get_confidence()

        if reset_offset:
            self.disp_grid.set_offset(0)

        self.disp_grid.update(rows, cols, class_, sorting, labelled, confidence)
        self.update_statistics()

    def update_statistics(self):
        # update bottom offset label
        offset = self.disp_grid.get_offset()
        max_ = self.disp_grid.get_max_samples(self.get_labelled(), self.get_class())
        self.offset_label.set_text(str(offset) + "/" + str(max_))

        # update statistics frame
        # remove old entries
        for child in self.vbox_statistics.get_children():
            self.vbox_statistics.remove(child)
            child.destroy()

        stats = self.disp_grid.get_statistics()

        for c, _ in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            hbox = Gtk.HBox()
            class_label = Gtk.Label(c + ": ")
            labelled = stats[c]['labelled']
            total = stats[c]['labelled'] + stats[c]['unlabelled']
            fraction = labelled/total
            str_label = Gtk.Label(str(labelled) + "/" + str(total) + "(" + str(round(100*fraction, 2)) + "%)")
            hbox.pack_start(class_label, expand=True, fill=True, padding=0)
            hbox.pack_start(str_label, expand=False, fill=False, padding=0)
            self.vbox_statistics.pack_start(hbox, expand=False, fill=False, padding=0)

            #frame = Gtk.Frame(label=stat['class'] + ": " + str(labelled) + "/" + str(total))
            pbar = Gtk.ProgressBar()
            self.vbox_statistics.pack_start(pbar, expand=False, fill=False, padding=0)
            #frame.add(pbar)
            pbar.set_fraction(fraction)
            #pbar.set_text(str(round(100 * fraction, 2)) + "%")
            pbar.show()
            #frame.show()


            hbox.show()
            class_label.show()
            str_label.show()

    def menuitem_response(self, event):
        if event == "file.open":
            dialog = Gtk.FileChooserDialog("Open..",
                                           None,
                                           Gtk.FileChooserAction.OPEN,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        else:
            dialog = Gtk.FileChooserDialog("Save..",
                                           None,
                                           Gtk.FileChooserAction.SAVE,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
            dialog.set_do_overwrite_confirmation(True)

        dialog.set_default_response(Gtk.ResponseType.OK)

        if event == "file.save" or event == "file.open":
            filter = Gtk.FileFilter()
            filter.set_name("Classification Results File")
            filter.add_pattern("*.pkl")
            dialog.add_filter(filter)

        if event == "file.export":
            filter = Gtk.FileFilter()
            filter.set_name("CSV")
            # Allow both lowercase and uppercase CSV extensions
            filter.add_pattern("*.csv")
            filter.add_pattern("*.CSV")
            dialog.add_filter(filter)

        filter = Gtk.FileFilter()
        filter.set_name("All files")
        filter.add_pattern("*")
        dialog.add_filter(filter)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            path = dialog.get_filename()
            dialog.destroy()

            if event == "file.open":
                self.disp_grid.open_db(path, self.class_combo)
                self.update_grid()
                self.window.set_title(path)
            elif event == "file.save":
                self.disp_grid.save_db(path)
                self.window.set_title(path)
            elif event == "file.export":
                self.disp_grid.export_db(path)
            else:
                print("whoops")
                raise NotImplemented

        dialog.destroy()

    def menuitem_save_response(self, event):
        if self.disp_grid.db_path:
            self.disp_grid.save_db(self.disp_grid.db_path)
        else:

            self.menuitem_response(event)

    def on_button_clicked(self, button, type):
        curr_labelled = self.disp_grid.get_current_num_labelled()
        if curr_labelled == 0 or self.get_labelled():
            offset = self.get_rows() * self.get_cols()
            if type == 'prev':
                offset *= -1
            self.disp_grid.add_offset(offset, self.get_labelled(), self.get_class())

        self.update_grid(reset_offset=False)

    def show_about(self, widget):
        about = Gtk.AboutDialog()
        about.set_program_name("Class Viewer")
        about.set_version("0.1")
        about.set_copyright("(c) 2017 TU Berlin, Communication Systems Group")
        about.set_comments("Simple tool to display and relabel classification results.")

        about.set_logo(GdkPixbuf.Pixbuf.new_from_xpm_data(logo_xpm()))

        about.run()
        about.destroy()

def main():
    Gtk.main()
    sys.exit(0)


def logo_xpm():
    return ["306 100 3 1",
            " 	c #9C0001",
            ".	c #0000BF",
            "+	c #None",
            "++++++++++                                                                              ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",
            "+++++++                                                                                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.++++++++++++++++++++++++++++++++++++++++",
            "++++++                                                                                      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++....+++++++++++++++++++++++++++++++++++++++",
            "++++                                                                                          ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++......++++++++++++++++++++++++++++++++++++++",
            "++++                                                                                           ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.......++++++++++++++++++++++++++++++++++++++",
            "+++                                                                                            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.........+++++++++++++++++++++++++++++++++++++",
            "++                                                                                              ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........++++++++++++++++++++++++++++++++++++++",
            "++                                                                                               ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.........++++++++++++++++++++++++++++++++++++++++",
            "+                                                                                                ++++++                                +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.........+++++++++++++++++++++++++++++++++++++++++",
            "+                                                                                                 +++++                                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.........+++++++.++++++++++++++++++++++++++++++++++",
            "+                                                                                                 +++++                                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++........++++++....+++++++++++++++++++++++++++++++++",
            "                                                                                                  +++++                                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.........++++.....+++++++++++++++++++++++++++++++++",
            "                                                                                                  +++++                               ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.........++.......++++++++++++++++++++++++++++++++",
            "                                                                                                   +++                 +++++++++++    ++++++++.......................++++++++++++++++++++++++++++++++++++++++......................++++......................++++..................++++++++.......................",
            "                                                                                                   +++                ++++++++++++    ++++++++.......................++++++++++++++++++++++++++++++++++++++++......................++++......................+++++...............++++++++++.......................",
            "                                                                                                   +++                ++++++++++++    ++++++++........................+++++++++++++++++++++++++++++++++++++++......................++++......................++++++.............+++++++++++.......................",
            "                                                                                                   +++                ++++            ++++++++.........................++++++++++++++++++++++++++++++++++++++......................++++......................++++++............++++++++++++.......................",
            "                                                                                                  ++++                +++             ++++++++..........................+++++++++++++++++++++++++++++++++++++......................++++......................+++++++..........+++++++..++++.......................",
            "                                                                                                  ++++                +++             ++++++++...........................++++++++++++++++++++++++++++++++++++......................++++......................++++++++........++++++....++++.......................",
            "+                                                                                                 ++++                 ++++           ++++++++...........................++++++++++++++++++++++++++++++++++++......................++++......................+++++++++........++++......+++.......................",
            "+                                                                                                 ++++                +++++++++++     ++++++++............................+++++++++++++++++++++++++++++++++++......................++++......................+++++++++.........++........++.......................",
            "+                                                                                                 ++++               ++++++++++++     ++++++++.............................++++++++++++++++++++++++++++++++++......................++++......................++++++++++..................++.......................",
            "++                                                                                                ++++               ++++++++++++     ++++++++..............................+++++++++++++++++++++++++++++++++......................++++......................+++++++++++................+++.......................",
            "++                                                                                                ++++                                ++++++++..............................+++++++++++++++++++++++++++++++++......................++++......................++++++++++++.............+++++.......................",
            "+++                                                                                               +++                 +              +++++++++...............................++++++++++++++++++++++++++++++++......................++++......................+++++++++++++...........++++++.......................",
            "++++                                                                                              +++            +++ ++++++++++++    +++++++++................................+++++++++++++++++++++++++++++++......................++++......................+++++++++++++..........+++++++.......................",
            "+++++                                                                                             +++            +++ ++++++++++++    +++++++++.................................++++++++++++++++++++++++++++++......................++++......................++++++++++++++.......+++++++++.......................",
            "++++++                                                                                            +++            +++ ++++++++++++    +++++++++..................................+++++++++++++++++++++++++++++......................++++......................+++++++++++++++.....++++++++++.......................",
            "+++++++                                                                                          ++++                                +++++++++..................................+++++++++++++++++++++++++++++......................++++......................++++++++++++++++...+++++++++++.......................",
            "+++++++++                                                                                        ++++                                +++++++++...................................++++++++++++++++++++++++++++......................++++......................+++++++++++++++++.++++++++++++.......................",
            "++++++++++++++ +++++++++++++++++++++++++++++++++++++++++++++++++++                               ++++            +++++++++++++++     +++++++++....................................+++++++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                              ++++            +++++++++++++++     +++++++++.....................................++++++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                              ++++            +++++++++++++++     +++++++++.....................................++++++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                               ++++                       +++      +++++++++......................................+++++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                             ++++++                               ++++                                +++++++++.......................................++++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                             ++++++                               ++++               +++             ++++++++++........................................+++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                             ++++++                               +++                ++++            ++++++++++.........................................++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                             ++++++                               +++                 +++++          ++++++++++.........................................++++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                             ++++++                               +++                ++++++++++++    ++++++++++..........................................+++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                            +++++++                               +++                ++++++++++++    ++++++++++...........................................++++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                            +++++++                              ++++                +++++++++++     ++++++++++............................................+++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++++                            +++++++                              ++++                                ++++++++++.............................................++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                             ++++++                               ++++                  +             ++++++++++.............................................++++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                             ++++++                               ++++                ++++   +++      ++++++++++..............................................+++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                             ++++++                               ++++                ++++   ++++     ++++++++++...............................................++++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                             ++++++                               ++++               ++++++   +++    +++++++++++................................................+++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                             ++++++                               ++++               +++ ++   +++    +++++++++++......................+.........................+++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                             ++++++                               ++++               +++ +++  +++    +++++++++++......................++.........................++++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                             ++++++                               +++                +++  +++ +++    +++++++++++......................++..........................+++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                            +++++++                               +++                +++  +++++++    +++++++++++......................+++..........................++++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                            +++++++                               +++                +++++++++++     +++++++++++......................++++..........................+++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                            ++++++                               ++++                 +++++++++      +++++++++++......................+++++.........................+++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++++                            ++++++                               ++++                  +++++++       +++++++++++......................++++++.........................++++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                             ++++++                               ++++                                +++++++++++......................++++++..........................+++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                             ++++++                               ++++                    ++         ++++++++++++......................+++++++..........................++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                             ++++++                               ++++                 ++++++++      ++++++++++++......................++++++++.........................++++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                             ++++++                               ++++                ++++++++++     ++++++++++++......................+++++++++.........................+++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                             ++++++                               ++++               +++++++++++     ++++++++++++......................+++++++++..........................++++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                             ++++++                               ++++               ++++    ++++    ++++++++++++......................++++++++++..........................+++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                            +++++++                               ++++               +++     ++++    ++++++++++++......................+++++++++++..........................++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                            +++++++                               +++                +++      +++    ++++++++++++......................++++++++++++.........................++++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                            ++++++                                +++                +++     ++++    ++++++++++++......................+++++++++++++.........................+++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                            ++++++                               ++++                +++     +++     ++++++++++++......................+++++++++++++..........................++......................++++......................++++++++++++++++++++++++++++++.......................",
            "+++++++++++++++++++++++++++++                            ++++++                               ++++            +++++++++ +++++     ++++++++++++......................++++++++++++++..........................+......................++++.......................+++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++                             ++++++                               ++++           +++++++++++++++     +++++++++++++......................+++++++++++++++................................................++++.......................+++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++                             ++++++                               ++++           ++++++++++++++      +++++++++++++......................++++++++++++++++...............................................++++.......................+++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++                             ++++++                               ++++              ++++++++++       +++++++++++++......................++++++++++++++++...............................................++++.......................+++++++++++++++++++++++++++++.......................",
            "++++++++++++++++++++++++++++                             ++++++                               ++++                               +++++++++++++......................+++++++++++++++++..............................................++++.......................+++++++++++++++++++++++++++++......................+",
            "++++++++++++++++++++++++++++                             ++++++                               ++++                               +++++++++++++......................++++++++++++++++++.............................................++++.......................+++++++++++++++++++++++++++++......................+",
            "++++++++++++++++++++++++++++                            +++++++                               ++++                               +++++++++++++......................+++++++++++++++++++............................................++++.......................++++++++++++++++++++++++++++.......................+",
            "++++++++++++++++++++++++++++                            +++++++                               ++++                               +++++++++++++......................++++++++++++++++++++...........................................++++........................+++++++++++++++++++++++++++.......................+",
            "++++++++++++++++++++++++++++                            +++++++                               ++++                               +++++++++++++......................++++++++++++++++++++...........................................+++++.......................+++++++++++++++++++++++++++.......................+",
            "++++++++++++++++++++++++++++                            ++++++                                +++                                +++++++++++++......................+++++++++++++++++++++..........................................+++++.......................+++++++++++++++++++++++++++.......................+",
            "++++++++++++++++++++++++++++                            ++++++                                +++                                +++++++++++++......................++++++++++++++++++++++.........................................+++++........................+++++++++++++++++++++++++........................+",
            "+++++++++++++++++++++++++++                             ++++++                               ++++                               ++++++++++++++......................+++++++++++++++++++++++........................................+++++........................+++++++++++++++++++++++++........................+",
            "+++++++++++++++++++++++++++                             ++++++                               ++++                               ++++++++++++++......................+++++++++++++++++++++++........................................+++++.........................+++++++++++++++++++++++.........................+",
            "+++++++++++++++++++++++++++                             ++++++                               ++++                               ++++++++++++++......................++++++++++++++++++++++++.......................................+++++..........................+++++++++++++++++++++.........................++",
            "+++++++++++++++++++++++++++                             ++++++                               ++++                               ++++++++++++++......................+++++++++++++++++++++++++......................................++++++..........................+++++++++++++++++++..........................++",
            "+++++++++++++++++++++++++++                             ++++++                               ++++                              +++++++++++++++......................++++++++++++++++++++++++++.....................................++++++...........................+++++++++++++++++...........................++",
            "+++++++++++++++++++++++++++                            +++++++                               ++++                              +++++++++++++++......................+++++++++++++++++++++++++++....................................++++++.............................+++++++++++++.............................++",
            "+++++++++++++++++++++++++++                            +++++++                               ++++                              +++++++++++++++......................+++++++++++++++++++++++++++....................................+++++++................................+++++................................+++",
            "+++++++++++++++++++++++++++                            ++++++++                              ++++                             ++++++++++++++++......................++++++++++++++++++++++++++++...................................+++++++.....................................................................+++",
            "+++++++++++++++++++++++++++                            ++++++++                              ++++                             ++++++++++++++++......................+++++++++++++++++++++++++++++..................................+++++++....................................................................++++",
            "+++++++++++++++++++++++++++                            ++++++++                              +++                              ++++++++++++++++......................++++++++++++++++++++++++++++++.................................++++++++...................................................................++++",
            "+++++++++++++++++++++++++++                            ++++++++                              +++                             +++++++++++++++++......................++++++++++++++++++++++++++++++.................................++++++++..................................................................+++++",
            "++++++++++++++++++++++++++                             ++++++++                             ++++                            ++++++++++++++++++......................+++++++++++++++++++++++++++++++................................+++++++++.................................................................+++++",
            "++++++++++++++++++++++++++                             +++++++++                            ++++                            ++++++++++++++++++......................++++++++++++++++++++++++++++++++...............................++++++++++...............................................................++++++",
            "++++++++++++++++++++++++++                             +++++++++                            ++++                           +++++++++++++++++++......................+++++++++++++++++++++++++++++++++..............................++++++++++..............................................................+++++++",
            "++++++++++++++++++++++++++                            +++++++++++                           ++++                          ++++++++++++++++++++......................++++++++++++++++++++++++++++++++++.............................+++++++++++.............................................................+++++++",
            "++++++++++++++++++++++++++                            +++++++++++                           ++++                         +++++++++++++++++++++......................++++++++++++++++++++++++++++++++++.............................++++++++++++...........................................................++++++++",
            "++++++++++++++++++++++++++                            ++++++++++++                          ++++                         +++++++++++++++++++++......................+++++++++++++++++++++++++++++++++++............................+++++++++++++.........................................................+++++++++",
            "++++++++++++++++++++++++++                            ++++++++++++                          ++++                        ++++++++++++++++++++++......................++++++++++++++++++++++++++++++++++++...........................++++++++++++++.......................................................++++++++++",
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                         ++++                      ++++++++++++++++++++++++......................+++++++++++++++++++++++++++++++++++++..........................+++++++++++++++.....................................................+++++++++++",
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                        ++++                     +++++++++++++++++++++++++......................+++++++++++++++++++++++++++++++++++++..........................+++++++++++++++++.................................................+++++++++++++",
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                       +++                     ++++++++++++++++++++++++++......................++++++++++++++++++++++++++++++++++++++.........................++++++++++++++++++...............................................++++++++++++++",
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                      +++                   ++++++++++++++++++++++++++++......................+++++++++++++++++++++++++++++++++++++++........................++++++++++++++++++++...........................................++++++++++++++++",
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                    +++                 ++++++++++++++++++++++++++++++......................++++++++++++++++++++++++++++++++++++++++.......................++++++++++++++++++++++.......................................++++++++++++++++++",
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                 ++++              +++++++++++++++++++++++++++++++++......................+++++++++++++++++++++++++++++++++++++++++......................+++++++++++++++++++++++++.................................+++++++++++++++++++++",
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++              ++++           +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++...........................++++++++++++++++++++++++",
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++          ++++      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.................+++++++++++++++++++++++++++++"]


if __name__ == '__main__':
    ClassViewer()
    main()
