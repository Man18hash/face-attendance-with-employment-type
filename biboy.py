import os
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import pandas as pd
from fpdf import FPDF
from pathlib import Path
import ctypes

# Base folder under %LOCALAPPDATA%\FaceAttendance
APPDATA_DIR     = Path(os.getenv("LOCALAPPDATA")) / "FaceAttendance"
DATASET_DIR     = APPDATA_DIR / "dataset"
ATTENDANCE_FILE = APPDATA_DIR / "attendance.csv"

# Create dirs if missing
APPDATA_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Make the AppData folder hidden (Windows only)
FILE_ATTRIBUTE_HIDDEN = 0x02
ctypes.windll.kernel32.SetFileAttributesW(str(APPDATA_DIR), FILE_ATTRIBUTE_HIDDEN)

# For face-attendance pop-up:
import cv2
import csv
import numpy as np
from PIL import Image, ImageTk
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ─────────────── Shared Globals ───────────────

ADMIN_EMAIL     = "admin"
ADMIN_PASSWORD  = "123"

COLOR_PRIMARY      = "#2C3E50"
COLOR_SECONDARY    = "#3498DB"
COLOR_LIGHT_BG     = "#ECF0F1"
COLOR_WHITE        = "#FFFFFF"
COLOR_BUTTON_RED   = "#E74C3C"
COLOR_BUTTON_GREEN = "#27AE60"

# ─────────────── AMS POP-UP ───────────────

mtcnn  = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def process_dataset(path):
    embeddings = {}
    for img_name in os.listdir(path):
        if img_name.lower().endswith(('.png','jpg','jpeg')):
            img = Image.open(os.path.join(path, img_name))
            face = mtcnn(img)
            if face is not None:
                emb = resnet(face.unsqueeze(0))
                embeddings[os.path.splitext(img_name)[0]] = emb.detach().numpy()
    return embeddings

embeddings_dict = process_dataset(DATASET_DIR)

def recognize_face(face_emb, db, threshold=0.9):
    best, dist = None, float('inf')
    for name, db_emb in db.items():
        d = np.linalg.norm(face_emb.detach().numpy() - db_emb)
        if d < dist:
            best, dist = name, d
    return (best, dist) if dist < threshold else (None, dist)

def mark_attendance(name, typ):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([name, ts, typ])
    messagebox.showinfo("Recorded", f"{typ} for {name} at {ts}")

class FaceAttendanceApp:
    COLOR_HEADER_BG = "#3498DB"
    COLOR_BTN_IN    = "#27AE60"
    COLOR_BTN_OUT   = "#E74C3C"
    COLOR_BTN_AGAIN = "#B0B0B0"

    def __init__(self, root):
        self.root = root
        self.current_name = None

        self.root.title("Attendance Monitoring System")
        self.root.geometry("1280x720")

        hdr = tk.Frame(root, bg=self.COLOR_HEADER_BG, height=60)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="ATTENDANCE MONITORING SYSTEM",
                 font=("Arial",24), fg="white", bg=self.COLOR_HEADER_BG).pack(side=tk.LEFT, padx=20)
        tk.Button(hdr, text="Back", font=("Arial",12),
                  command=self.on_close).pack(side=tk.RIGHT, padx=20)

        self.dt_label = tk.Label(root,
            text=datetime.datetime.now().strftime("%B %d, %Y   %I:%M %p"),
            font=("Arial",14))
        self.dt_label.pack(pady=10)
        self.update_clock()

        # ─── Video preview frame ───
        self.video_frame = tk.Frame(root, bg="#DDDDDD", width=800, height=300)
        self.video_frame.pack(padx=20, pady=10)
        self.video_frame.pack_propagate(False)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.name_label = tk.Label(root, text="No face detected",
                                   font=("Arial",16))
        self.name_label.pack(pady=10)

        btn_f = tk.Frame(root)
        btn_f.pack(pady=20)
        tk.Button(btn_f, text="Time In", bg=self.COLOR_BTN_IN, fg="white",
                  font=("Arial",14), width=12,
                  command=lambda: self.record("Time In")).grid(row=0, column=0, padx=15)
        tk.Button(btn_f, text="Time Out", bg=self.COLOR_BTN_OUT, fg="white",
                  font=("Arial",14), width=12,
                  command=lambda: self.record("Time Out")).grid(row=0, column=1, padx=15)
        tk.Button(btn_f, text="Again", bg=self.COLOR_BTN_AGAIN, fg="black",
                  font=("Arial",14), width=12,
                  command=self.reset_detection).grid(row=0, column=2, padx=15)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.update_video()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_clock(self):
        self.dt_label.config(
            text=datetime.datetime.now().strftime("%B %d, %Y   %I:%M %p"))
        self.root.after(60000, self.update_clock)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640,360))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            boxes, _ = mtcnn.detect(pil)
            if boxes is not None and len(boxes):
                face = mtcnn(pil)
                if face is not None:
                    emb = resnet(face.unsqueeze(0))
                    name, _ = recognize_face(emb, embeddings_dict)
                    if name:
                        self.current_name = name
                        self.name_label.config(text=f"{name} detected")
                        x1,y1,x2,y2 = map(int, boxes[0])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    else:
                        self.name_label.config(text="Unknown face")
                else:
                    self.name_label.config(text="Face not aligned")
            else:
                self.current_name = None
                self.name_label.config(text="No face detected")

            imgtk = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(10, self.update_video)

    def record(self, typ):
        if not self.current_name:
            messagebox.showwarning("No one", "No recognized face to record.")
            return
        mark_attendance(self.current_name, typ)

    def reset_detection(self):
        self.current_name = None
        self.name_label.config(text="No face detected")

    def on_close(self):
        self.cap.release()
        self.root.destroy()

# ─────────────── USER REGISTRATION POP-UP ───────────────

class UserRegistration:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Monitoring System")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLOR_LIGHT_BG)

        header = tk.Frame(self.root, bg=COLOR_SECONDARY, height=80)
        header.pack(fill=tk.X); header.pack_propagate(False)
        tk.Label(header, text="ATTENDANCE MONITORING SYSTEM",
                 font=("Arial",20,"bold"),
                 bg=COLOR_SECONDARY, fg=COLOR_WHITE).pack(side=tk.LEFT, padx=20)
        tk.Button(header, text="Back", font=("Arial",12),
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  relief=tk.FLAT, command=self.root.destroy).pack(side=tk.RIGHT, padx=10, pady=10)

        self.video_frame = tk.Frame(self.root, bg="#DDDDDD", width=800, height=300)
        self.video_frame.pack(pady=20); self.video_frame.pack_propagate(False)
        self.video_label = tk.Label(self.video_frame); self.video_label.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(self.root, bg=COLOR_LIGHT_BG)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Take", font=("Arial",12),
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  width=12, command=self.capture).grid(row=0, column=0, padx=20)
        tk.Button(btn_frame, text="Retake", font=("Arial",12),
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  width=12, command=self.retake).grid(row=0, column=1, padx=20)

        entry_frame = tk.Frame(self.root, bg=COLOR_LIGHT_BG)
        entry_frame.pack(pady=20)
        tk.Label(entry_frame, text="Name:", font=("Arial",14), bg=COLOR_LIGHT_BG).grid(row=0, column=0, sticky=tk.E, padx=10, pady=5)
        self.name_entry = tk.Entry(entry_frame, font=("Arial",14), width=30)
        self.name_entry.grid(row=0, column=1, pady=5)
        tk.Label(entry_frame, text="Employee Type:", font=("Arial",14), bg=COLOR_LIGHT_BG).grid(row=1, column=0, sticky=tk.E, padx=10, pady=5)
        self.type_var = tk.StringVar()
        self.type_dropdown = ttk.Combobox(entry_frame, textvariable=self.type_var, font=("Arial",14),
                                          values=["Teaching","Non-Teaching","Job-Order"],
                                          state="readonly", width=28)
        self.type_dropdown.current(0); self.type_dropdown.grid(row=1, column=1, pady=5)
        tk.Label(entry_frame, text="Position:", font=("Arial",14), bg=COLOR_LIGHT_BG).grid(row=2, column=0, sticky=tk.E, padx=10, pady=5)
        self.position_entry = tk.Entry(entry_frame, font=("Arial",14), width=30)
        self.position_entry.grid(row=2, column=1, pady=5)

        tk.Button(self.root, text="Save", font=("Arial",14),
                  bg=COLOR_BUTTON_GREEN, fg=COLOR_WHITE,
                  width=40, command=self.save).pack(pady=20)
        tk.Button(self.root, text="User List", font=("Arial",14),
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  width=40, command=self.open_user_list).pack(pady=5)

        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = None
        self.captured_image = None
        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)

    def update_video(self):
        if self.captured_image is None:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def capture(self):
        if self.current_frame is not None:
            self.captured_image = self.current_frame.copy()
            cv2image = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            messagebox.showwarning("Capture Error", "No frame available")

    def retake(self):
        self.captured_image = None

    def save(self):
        if self.captured_image is None:
            messagebox.showwarning("No Image", "Capture first.")
            return
        name = self.name_entry.get().strip()
        emp_type = self.type_var.get().strip()
        pos = self.position_entry.get().strip()
        if not name or not emp_type or not pos:
            messagebox.showwarning("Missing", "Fill all fields.")
            return
        fn = f"{name}, {emp_type}, {pos}.jpg"
        path = DATASET_DIR / fn
        cv2.imwrite(str(path), self.captured_image)
        messagebox.showinfo("Saved", f"Image saved to {path}")
        self.captured_image = None
        self.name_entry.delete(0, tk.END)
        self.type_dropdown.current(0)
        self.position_entry.delete(0, tk.END)

    def open_user_list(self):
        popup = tk.Toplevel(self.root)
        UserListApp(popup)

    def cleanup(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.root.destroy()

# ─────────────── USER LIST POP-UP ───────────────

class UserListApp:
    def __init__(self, root):
        self.root = root
        self.root.title("User List")
        self.root.geometry("800x600")
        self.root.configure(bg=COLOR_LIGHT_BG)

        header = tk.Frame(self.root, bg=COLOR_SECONDARY, height=50)
        header.pack(fill=tk.X)
        tk.Label(header, text="Registered Users", font=("Arial",20,"bold"),
                 bg=COLOR_SECONDARY, fg=COLOR_WHITE).pack(side=tk.LEFT, padx=20)
        tk.Button(header, text="Back", font=("Arial",12),
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  command=self.root.destroy).pack(side=tk.RIGHT, padx=20, pady=10)

        cols = ("Name","Employment Type","Position")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings")
        for c in cols:
            w = 250 if c=="Name" else 200
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        btn_frame = tk.Frame(self.root, bg=COLOR_LIGHT_BG)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Edit Selected", font=("Arial",14),
                  bg="#F39C12", fg=COLOR_WHITE,
                  command=self.edit_selected).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Delete Selected", font=("Arial",14),
                  bg=COLOR_BUTTON_RED, fg=COLOR_WHITE,
                  command=self.delete_selected).pack(side=tk.LEFT, padx=10)

        self.load_users()

    def load_users(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for fname in sorted(os.listdir(DATASET_DIR)):
            if not fname.lower().endswith(('.png','.jpg','jpeg')):
                continue
            base = os.path.splitext(fname)[0]
            parts = [p.strip() for p in base.split(',')]
            name = parts[0] if len(parts)>0 else ''
            et   = parts[1] if len(parts)>1 else ''
            pos  = parts[2] if len(parts)>2 else ''
            self.tree.insert('', tk.END, iid=fname, values=(name, et, pos))

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No Selection","Select a user.")
            return
        if not messagebox.askyesno("Confirm","Delete selected?"):
            return
        for iid in sel:
            try:
                os.remove(DATASET_DIR / iid)
            except Exception as e:
                messagebox.showerror("Error", f"{e}")
        self.load_users()

    def edit_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No Selection","Select a user.")
            return
        old = sel[0]
        vals = self.tree.item(old)['values']
        old_name, old_type, old_pos = vals
        ext = os.path.splitext(old)[1]
        ew = tk.Toplevel(self.root)
        ew.title("Edit User"); ew.geometry("350x250")

        tk.Label(ew, text="Name:", font=("Arial",12)).pack(pady=5)
        ne = tk.Entry(ew, font=("Arial",12)); ne.insert(0, old_name); ne.pack(pady=5)
        tk.Label(ew, text="Employee Type:", font=("Arial",12)).pack(pady=5)
        tv = tk.StringVar(value=old_type)
        td = ttk.Combobox(ew, textvariable=tv, font=("Arial",12),
                          values=["Teaching","Non-Teaching","Job-Order"],
                          state="readonly", width=24); td.pack(pady=5)
        tk.Label(ew, text="Position:", font=("Arial",12)).pack(pady=5)
        pe = tk.Entry(ew, font=("Arial",12)); pe.insert(0, old_pos); pe.pack(pady=5)

        def save_changes():
            nn = ne.get().strip(); nt = tv.get().strip(); np = pe.get().strip()
            if not nn:
                messagebox.showerror("Invalid","Name empty.")
                return
            new = f"{nn}, {nt}, {np}{ext}"
            try:
                os.rename(DATASET_DIR / old, DATASET_DIR / new)
                ew.destroy(); self.load_users()
            except Exception as e:
                messagebox.showerror("Error", f"{e}")

        tk.Button(ew, text="Save", font=("Arial",12),
                  bg=COLOR_BUTTON_GREEN, fg=COLOR_WHITE,
                  command=save_changes).pack(pady=15)

# ─────────────── ADMIN PANEL ───────────────

class AdminApp:
    def __init__(self, root):
        self.root  = root
        self.popup = None
        self.displayed_df = pd.DataFrame(
            columns=["NameField","Timestamp","AttendType","EmpType"])

        self.root.title("Admin Login")
        self.root.geometry("800x500")
        self.root.configure(bg=COLOR_PRIMARY)
        self.create_header()
        self.create_login_frame()

    def create_header(self):
        hdr = tk.Frame(self.root, bg=COLOR_SECONDARY, height=50)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="AMS", font=("Arial",24,"bold"),
                 fg=COLOR_WHITE, bg=COLOR_SECONDARY).pack(side=tk.LEFT, padx=20)
        tk.Label(hdr, text="ATTENDANCE MONITORING SYSTEM",
                 font=("Arial",14), fg=COLOR_WHITE,
                 bg=COLOR_SECONDARY).pack(side=tk.LEFT)

    def create_login_frame(self):
        frm = tk.Frame(self.root, bg=COLOR_LIGHT_BG, padx=30, pady=30)
        frm.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(frm, text="Admin Login", font=("Arial",18,"bold"),
                 bg=COLOR_LIGHT_BG).grid(row=0, column=0, columnspan=2, pady=(0,20))
        tk.Label(frm, text="Email:", bg=COLOR_LIGHT_BG).grid(row=1, column=0, sticky=tk.W)
        self.email_entry = tk.Entry(frm); self.email_entry.grid(row=1, column=1, pady=5)
        tk.Label(frm, text="Password:", bg=COLOR_LIGHT_BG).grid(row=2, column=0, sticky=tk.W)
        self.password_entry = tk.Entry(frm, show="*"); self.password_entry.grid(row=2, column=1, pady=5)

        tk.Button(frm, text="Login", bg=COLOR_BUTTON_RED, fg=COLOR_WHITE, font=("Arial",12),
                  command=self.login).grid(row=3, column=0, columnspan=2, pady=15, ipadx=20)

    def login(self):
        if (self.email_entry.get()==ADMIN_EMAIL and
            self.password_entry.get()==ADMIN_PASSWORD):
            self.show_admin_panel()
        else:
            messagebox.showerror("Login Failed","Invalid credentials.")

    def show_admin_panel(self):
        for w in self.root.winfo_children():
            w.destroy()

        self.root.title("Admin Panel"); self.root.configure(bg=COLOR_PRIMARY)

        side = tk.Frame(self.root, bg=COLOR_SECONDARY, width=180)
        side.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(side, text="AMS", font=("Arial",20,"bold"),
                 fg=COLOR_WHITE, bg=COLOR_SECONDARY).pack(pady=20)
        tk.Button(side, text="Attendance", width=16,
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  command=self.open_attendance).pack(pady=5)
        tk.Button(side, text="User", width=16,
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  command=self.open_user).pack(pady=5)
        tk.Button(side, text="User List", width=16,
                  bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  command=self.open_userlist).pack(pady=5)
        tk.Button(side, text="Logout", width=16,
                  bg=COLOR_BUTTON_RED, fg=COLOR_WHITE,
                  command=self.root.destroy).pack(side=tk.BOTTOM, pady=20)

        main = tk.Frame(self.root, bg=COLOR_LIGHT_BG)
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(main, text="Attendance Records",
                 font=("Arial",16,"bold"), bg=COLOR_LIGHT_BG).pack(pady=10)

        cols = ("NameField","Timestamp","AttendType","EmpType")
        self.tree = ttk.Treeview(main, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=180, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Filters
        f = tk.Frame(main, bg=COLOR_LIGHT_BG); f.pack(pady=5)
        tk.Label(f, text="Start Date:", bg=COLOR_LIGHT_BG).grid(row=0, column=0)
        self.start_cal = DateEntry(f, date_pattern='yyyy-mm-dd')
        self.start_cal.grid(row=0, column=1, padx=5)
        tk.Label(f, text="End Date:", bg=COLOR_LIGHT_BG).grid(row=0, column=2)
        self.end_cal = DateEntry(f, date_pattern='yyyy-mm-dd')
        self.end_cal.grid(row=0, column=3, padx=5)
        tk.Label(f, text="Employment Type:", bg=COLOR_LIGHT_BG).grid(row=0, column=4, padx=(20,0))
        self.type_filter = ttk.Combobox(f, values=["All","Teaching","Non-Teaching","Job-Order"],
                                        state="readonly", width=15)
        self.type_filter.current(0); self.type_filter.grid(row=0, column=5, padx=5)
        tk.Button(f, text="Filter", bg=COLOR_SECONDARY, fg=COLOR_WHITE,
                  command=self.apply_filters).grid(row=0, column=6, padx=5)
        tk.Button(f, text="Reset", bg=COLOR_BUTTON_GREEN, fg=COLOR_WHITE,
                  command=self.load_attendance).grid(row=0, column=7, padx=5)

        # Exports
        ef = tk.Frame(main, bg=COLOR_LIGHT_BG); ef.pack(pady=10)
        tk.Button(ef, text="Export CSV", bg=COLOR_BUTTON_RED, fg=COLOR_WHITE,
                  command=self.export_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(ef, text="Export Excel", bg=COLOR_BUTTON_RED, fg=COLOR_WHITE,
                  command=self.export_excel).pack(side=tk.LEFT, padx=5)
        tk.Button(ef, text="Export PDF", bg=COLOR_BUTTON_RED, fg=COLOR_WHITE,
                  command=self.export_pdf).pack(side=tk.LEFT, padx=5)

        self.load_attendance()

    def close_popup(self):
        if self.popup and tk.Toplevel.winfo_exists(self.popup):
            self.popup.destroy(); self.popup = None

    def open_attendance(self):
        self.close_popup()
        self.popup = tk.Toplevel(self.root)
        FaceAttendanceApp(self.popup)

    def open_user(self):
        self.close_popup()
        self.popup = tk.Toplevel(self.root)
        UserRegistration(self.popup)

    def open_userlist(self):
        self.close_popup()
        self.popup = tk.Toplevel(self.root)
        UserListApp(self.popup)

    def load_attendance(self):
        if not ATTENDANCE_FILE.exists():
            self.displayed_df = pd.DataFrame(
                columns=["NameField","Timestamp","AttendType","EmpType"])
        else:
            df = pd.read_csv(ATTENDANCE_FILE, header=None,
                             names=["NameField","Timestamp","AttendType"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["EmpType"] = df["NameField"].apply(
                lambda x: x.split(",")[1].strip() if "," in x else "")
            self.displayed_df = df.copy()
        self._refresh_tree(self.displayed_df)

    def apply_filters(self):
        df = self.displayed_df.copy()
        start = pd.Timestamp(self.start_cal.get_date())
        end   = pd.Timestamp(self.end_cal.get_date()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
        sel = self.type_filter.get()
        if sel != "All":
            df = df[df["EmpType"] == sel]
        self._refresh_tree(df)

    def _refresh_tree(self, df):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for _, row in df.iterrows():
            ts = row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            self.tree.insert("", tk.END, values=(
                row["NameField"], ts, row["AttendType"], row["EmpType"]))

    def export_csv(self):
        if self.displayed_df.empty:
            messagebox.showwarning("No Data", "Nothing to export."); return
        path = Path.home() / "Desktop" / f"Attendance_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        self.displayed_df.to_csv(path, index=False)
        messagebox.showinfo("Export", f"CSV saved to {path}")

    def export_excel(self):
        if self.displayed_df.empty:
            messagebox.showwarning("No Data", "Nothing to export."); return
        path = Path.home() / "Desktop" / f"Attendance_{datetime.datetime.now():%Y%m%d_%H%M%S}.xlsx"
        self.displayed_df.to_excel(path, index=False)
        messagebox.showinfo("Export", f"Excel saved to {path}")

    def export_pdf(self):
        if self.displayed_df.empty:
            messagebox.showwarning("No Data", "Nothing to export.")
            return

        filename = f"Attendance_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf"
        path = Path.home() / "Desktop" / filename

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        effective_width = pdf.w - 2 * pdf.l_margin
        col_w = effective_width / len(self.tree["columns"])

        for col in ["NameField","Timestamp","AttendType","EmpType"]:
            pdf.cell(col_w, 10, col, border=1, align='C')
        pdf.ln(10)

        for _, row in self.displayed_df.iterrows():
            pdf.cell(col_w, 8, row["NameField"], border=1)
            pdf.cell(col_w, 8, row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"), border=1)
            pdf.cell(col_w, 8, row["AttendType"], border=1)
            pdf.cell(col_w, 8, row["EmpType"], border=1)
            pdf.ln(8)

        pdf.output(str(path))
        messagebox.showinfo("Export", f"PDF saved to {path}")

if __name__ == "__main__":
    root = tk.Tk()
    AdminApp(root)
    root.mainloop()
