function submitImage(imagePath) {
    const form = document.getElementById('imageForm');

    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'selected_image';
    input.value = imagePath;
    form.appendChild(input);

    form.submit();
}
