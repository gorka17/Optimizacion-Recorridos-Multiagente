import cv2

# Coordenadas de las ciudades
city_coordinates = {
    'Madrid': (256, 221), 'Barcelona': (495, 168), 'Valencia': (391, 272),
    'Sevilla': (162, 388), 'Bilbao': (299, 63), 'A Coruna': (69, 66),
    'Leon': (162, 96), 'Salamanca': (183, 184), 'Murcia': (356, 360),
    'Badajoz': (132, 293), 'Zaragoza': (387, 154), 'C. Real': (284, 276)
}

# Cargar la imagen base
img = cv2.imread('Mapa_inicial.jpg')

# Dibujar círculos y etiquetas en las ciudades
for city, coordinates in city_coordinates.items():
    # Dibujar el círculo (borde negro, relleno blanco)
    cv2.circle(img, center=coordinates, radius=5, color=(0, 0, 0), thickness=5)
    cv2.circle(img, center=coordinates, radius=5, color=(255, 255, 255), thickness=2)

    # Añadir el nombre de la ciudad
    text_position = (coordinates[0] + 10, coordinates[1] - 10)  # Desplazar el texto ligeramente
    cv2.putText(img, city, text_position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                0.75, (0, 0, 0), 1)

# Mostrar la imagen en un bucle
while cv2.waitKey(1) == -1:
    cv2.imshow('Mapa dibujado', img)

cv2.imwrite('Mapa_nuevo.jpg', img)

cv2.destroyAllWindows()
