$(document).ready(function () {
    // Form Submission Handling
    $('#bookingForm').on('submit', function (e) {
      e.preventDefault();
      if (this.checkValidity()) {
        alert('Booking successful! Thank you for choosing our service.');
        // You can add AJAX call here to submit the form data
      } else {
        alert('Please fill out all fields correctly.');
      }
    });
  });
