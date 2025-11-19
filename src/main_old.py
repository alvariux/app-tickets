import flet as ft


def main(page: ft.Page): 
    def btn_login_clicked(e):        
        usuario = txt_usuario.value
        contrasena = txt_contrasena.value        

        login_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Información de Inicio de Sesión"),
            content=ft.Text(f"Usuario: {usuario}\nContraseña: {contrasena}"),
            actions=[ft.TextButton("Cerrar", on_click=lambda e: page.close(login_dialog))],
        )
        page.open(login_dialog)
        

    lbl_app = ft.Text("Bienvenido a la APP")
    txt_usuario = ft.TextField(label="Usuario", width=300)
    txt_contrasena = ft.TextField(label="Contraseña", width=300, password=True, can_reveal_password=True)
    btn_login = ft.ElevatedButton(text="Iniciar Sesión", width=150, on_click=lambda e: btn_login_clicked(e)) 
    
    page.add(
        ft.SafeArea(
            ft.Container(
                content=ft.Column(
                    controls=[
                        lbl_app,
                        ft.Divider(height=20),                        
                        txt_usuario,
                        txt_contrasena,
                        ft.Divider(height=20),
                        btn_login,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                ),
                alignment=ft.alignment.center,
            ),
            expand=True,
        )
    )


ft.app(main)
