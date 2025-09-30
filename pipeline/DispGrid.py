# ---------------------------------------------------------
# Class Viewer
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Written by Erik Bochinski
# ---------------------------------------------------------


import random
import pickle
from time import sleep
import os
import csv

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

import cairo


# works but probably not the best way
def gen_scaled_surface(pixbuf, scale):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(pixbuf.get_width()*scale),
                                 int(pixbuf.get_height()*scale))
    cr = cairo.Context(surface)
    cr.scale(scale, scale)
    Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
    cr.paint()
    return surface


class DrawingButton(Gtk.DrawingArea):
    def __init__(self, sample, base_path, options, origin_labelled):
        Gtk.DrawingArea.__init__(self)

        self.sample = sample
        #self.path = base_path + "/" + self.sample['paths']
        self.path = self.sample['paths']
        if 'label' in sample:
            self.class_ = sample['label']
        else:
            self.class_ = sample['y_predicted']
        self.initial_class = sample['y_predicted']
        self.options = options
        self.origin_labelled = origin_labelled

        self.pb = GdkPixbuf.Pixbuf.new_from_file(self.path)

        self.connect('draw', self.drawingarea_draw)
        self.connect('button-press-event', self.on_press)
        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)

    def drawingarea_draw(self, widget, cr):

        w = widget.get_allocated_width()
        h = widget.get_allocated_height()

        width_ratio = float(w) / float(self.pb.get_width())
        height_ratio = float(h) / float(self.pb.get_height())

        scale_xy = min(height_ratio, width_ratio)


        #Gdk.cairo_set_source_pixbuf(cr, self.pb, 0, 0)
        cr.set_source_surface(gen_scaled_surface(self.pb, scale_xy))
        cr.paint()

        cr.set_line_width(5)

        if 'label' not in self.sample:
            cr.set_source_rgb(0, 0, 1)
        else:
            if self.class_ == self.initial_class:
                cr.set_source_rgb(0, 1, 0)
            else:
                cr.set_source_rgb(1, 0, 0)

        cr.move_to(6, 16)
        cr.set_font_size(15)
        cr.show_text(self.options[self.class_])
        #cr.paint()

        cr.rectangle(0, 0, w, h)
        cr.stroke()

    def on_press(self, widget, event):
        if event.type == Gdk.EventType.BUTTON_PRESS:
            # TODO hardcoded for junk, could be made to a parameter
            if event.button == 3 and "junk" in self.options:  # right click
                self.class_ = self.options.index("junk")
            else:
                self.class_ = (self.class_ + 1) % len(self.options)
            self.sample['label'] = self.class_

            self.queue_draw()

    def accept(self):
        self.sample['label'] = self.class_
        self.queue_draw()


class DispGrid:
    def __init__(self, parent, rows, cols, class_, sorting, labelled, confidence):
        self.db = None
        self.db_path = None
        self.labelled = {}
        self.unlabelled = {}
        self.last_sorting = None
        self.last_confidence = None
        self.base_path = None
        self.disp_grid = None
        self.parent = parent

        self.offset = 0

        self.dareas = []

        self.update(rows, cols, class_, sorting, labelled, confidence)

    def update(self, rows, cols, class_, sorting, labelled, confidence):
        if self.disp_grid:
            self.disp_grid.destroy()
        self.disp_grid = Gtk.Grid(row_homogeneous=True, column_homogeneous=True)
        self.disp_grid.set_hexpand(True)
        self.disp_grid.set_vexpand(True)
        self.parent.attach(self.disp_grid, 0, 0, 1, 1)

        #if not self.db:
        #    return

        for darea in self.dareas:
            if not darea.origin_labelled and 'label' in darea.sample:
                darea_class = darea.options[darea.sample['y_predicted']]

                idx = next(index for (index, d) in enumerate(self.unlabelled[darea_class])
                           if d['paths'] == darea.sample['paths'])
                self.labelled[darea_class].append(self.unlabelled[darea_class].pop(idx))

                idx = next(index for (index, d) in enumerate(self.unlabelled["All"])
                           if d['paths'] == darea.sample['paths'])
                self.labelled["All"].append(self.unlabelled["All"].pop(idx))

        self.dareas = []
        samples = self.get_next(rows*cols, class_, sorting, labelled, confidence)
        for row in range(rows):
            for col in range(cols):
                sample = samples.pop()
                if sample:
                    darea = DrawingButton(sample, self.base_path, self.db['classes'], labelled)
                    self.disp_grid.attach(darea, col, row, 1, 1)
                    self.dareas.append(darea)
                # else:
                #     if (row + col) % 2:
                #         path = "nue.png"
                #     else:
                #         path = "tub.png"
                #     darea = DrawingButton({'paths': path, 'y_predicted': random.choice([0,1])}, ".", ["NUE", "TUB"])
                #     self.disp_grid.attach(darea, row, col, 1, 1)
                #     self.dareas.append(darea)

        self.disp_grid.show_all()

    def get_current_num_labelled(self):
        return sum([1 if 'label' in darea.sample else 0 for darea in self.dareas])

    def accept(self, button):
        for darea in self.dareas:
            darea.accept()

    def open_db(self, path, combo):
        self.db_path = path
        self.base_path = os.path.dirname(path)
        message = Gtk.MessageDialog(type=Gtk.MessageType.INFO)
        message.set_markup("Opening File...")
        message.show()

        sleep(0.01)  # needed for events pending to work
        while Gtk.events_pending():
            Gtk.main_iteration()

        with open(path, 'rb') as fd:
            self.db = pickle.load(fd)

        # TODO only for testing
        #self.db['samples'] = self.db['samples'][:100]

        # TODO ask for this dir
        print("checking")
        #base_dir = os.path.dirname(path)
        missing_dirs = {}
        for sample in self.db['samples']:
            if not os.path.exists(sample['paths']):
                missing_dir = os.path.dirname(sample['paths'])
                if missing_dir in missing_dirs:
                    missing_dirs[missing_dir] += 1
                else:
                    missing_dirs[missing_dir] = 1

                #print(base_dir+"/"+sample['paths'])

        # load db into labelled and unlabelled dicts
        self.labelled = {'All': [sample for sample in self.db['samples'] if 'label' in sample]}
        self.unlabelled = {'All': [sample for sample in self.db['samples'] if 'label' not in sample]}
        self.last_sorting = None  # force resort at next update

        for class_name in self.db['classes']:
            class_id = self.db['classes'].index(class_name)

            # TODO one for instead of two list comprehensions
            self.labelled[class_name] = [sample for sample in self.db['samples'] if
                                         sample['y_predicted'] == class_id and 'label' in sample]
            self.unlabelled[class_name] = [sample for sample in self.db['samples'] if
                                           sample['y_predicted'] == class_id and 'label' not in sample]

        class_selection_store = Gtk.ListStore(str)
        class_selection_store.append(["All"])
        for class_ in self.db['classes']:
            class_selection_store.append([class_])

        combo.set_model(class_selection_store)
        combo.set_entry_text_column(0)
        combo.set_active(0)

        message.destroy()

        if missing_dirs:
            dialog = Gtk.Dialog(title="Files Missing!", buttons=(Gtk.STOCK_OK, Gtk.ResponseType.OK))
            box = dialog.get_content_area()
            label1 = Gtk.Label("Warning, not all image files are available. Make sure that the opened\n"
                               "results file is located at the root of the following directories: ")
            box.add(label1)
            label2 = Gtk.Label(" ")
            box.add(label2)

            for missing_dir in missing_dirs:
                label = Gtk.Label(missing_dir + " : " + str(missing_dirs[missing_dir]) + " files missing!")
                box.add(label)

            dialog.show_all()
            dialog.run()
            dialog.destroy()

    def save_db(self, path):
        message = Gtk.MessageDialog(type=Gtk.MessageType.INFO)
        message.set_markup("Saving to File...")
        message.show()

        sleep(0.01)  # needed for events pending to work
        while Gtk.events_pending():
            Gtk.main_iteration()

        with open(path, 'wb') as fd:
            pickle.dump(self.db, fd)

        message.destroy()

        self.db_path = path
        # TODO ask for dir
        #self.base_path = os.path.dirname(path)

    def sort(self, data, sorting, confidence):
        for c in data:
            if sorting == "Rand":
                random.shuffle(data[c])
            elif sorting == "EN":
                data[c] = sorted(data[c], key=lambda x: x['HC_en'], reverse=not confidence)
            elif sorting == "MS":
                data[c] = sorted(data[c], key=lambda x: x['HC_ms'], reverse=confidence)
            elif sorting == "LC":
                data[c] = sorted(data[c], key=lambda x: x['HC_lc'], reverse=confidence)
            else:
                print("whoops")
                raise LookupError

        return data

    def add_offset(self, offset, labelled, class_):
        self.offset += offset
        self.offset = max(0, self.offset)
        self.offset = min(self.offset, self.get_max_samples(labelled, class_))

    def set_offset(self, offset):
        self.offset = offset

    def get_offset(self):
        return self.offset

    def get_max_samples(self, labelled, class_):
        try:
            if labelled:
                return len(self.labelled[class_])
            else:
                return len(self.unlabelled[class_])
        except KeyError:
            return 0

    def get_next(self, num_samples, class_, sorting, labelled, confidence):
        if not self.labelled or not self.unlabelled:
            return [None] * num_samples

        samples = []

        if sorting != self.last_sorting or confidence != self.last_confidence:
            self.last_sorting = sorting
            self.last_confidence = confidence
            self.labelled = self.sort(self.labelled, sorting, confidence)
            self.unlabelled = self.sort(self.unlabelled, sorting, confidence)

        for i in range(num_samples):
            try:
                idx = self.offset + i
                if labelled:
                    samples.append(self.labelled[class_][idx])
                else:
                    samples.append(self.unlabelled[class_][idx])
            except IndexError:
                samples.append(None)

        return samples

    def get_statistics(self):
        stats = {}
        for c in self.labelled:
            assert c in self.labelled
            stats[c] = {'labelled': len(self.labelled[c]), 'unlabelled': len(self.unlabelled[c]),
                        'total': len(self.labelled[c]) + len(self.unlabelled[c])}

        return stats

    def export_db(self, path):
        fieldnames = ['path', 'prediction', 'label']
        with open(path, 'w') as fd:
            dict_writer = csv.DictWriter(fd, fieldnames)
            dict_writer.writeheader()
            if not self.db:
                # TODO waring message?
                print("no data to export available!")
            else:
                for sample in self.db['samples']:
                    row = {'path': sample['paths'],
                           'prediction': self.db['classes'][sample['y_predicted']],
                           'label': self.db['classes'][sample['label']] if 'label' in sample else ''}
                    dict_writer.writerow(row)
