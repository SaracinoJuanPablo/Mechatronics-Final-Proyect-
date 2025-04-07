from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from plyer import camera
from android.permissions import request_permissions, Permission  # <-- Importar aquí


class MyApp(App):
    def on_start(self):
        # Solicitar permisos al iniciar la app (¡esto es crítico!)
        request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
    def take_photo(self):
        try:
            # Toma una foto y guarda en el almacenamiento interno
            camera.take_picture(
                filename='/sdcard/DCIM/photo.jpg',
                on_complete=lambda *args: print("Foto tomada!")
            )
        except Exception as e:
            print(f"Error: {e}")
    def get_photo_path(self):
        # Usa una ruta compatible con Android
        from os.path import join
        from android.storage import app_storage_path
        return join(app_storage_path(), "photo.jpg")

class MainLayout(BoxLayout):
    pass

if __name__ == '__main__':
    MyApp().run()