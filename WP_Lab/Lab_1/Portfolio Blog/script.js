document.addEventListener('DOMContentLoaded', () => {
  const darkModeToggle = document.getElementById('darkModeToggle');
  const body = document.body;

  // Check for saved dark mode preference
  if (localStorage.getItem('darkMode') === 'enabled') {
      body.classList.add('dark-mode');
      darkModeToggle.textContent = '☀️';
  }

  // Dark mode toggle functionality
  darkModeToggle.addEventListener('click', () => {
      body.classList.toggle('dark-mode');
      if (body.classList.contains('dark-mode')) {
          localStorage.setItem('darkMode', 'enabled');
          darkModeToggle.textContent = '☀️';
      } else {
          localStorage.setItem('darkMode', null);
          darkModeToggle.textContent = '🌙';
      }
  });

  // Smooth scrolling for navigation links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
          e.preventDefault();
          document.querySelector(this.getAttribute('href')).scrollIntoView({
              behavior: 'smooth'
          });
      });
  });

  // Simple form submission (you'd typically send this data to a server)
  const contactForm = document.getElementById('contactForm');
  contactForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const formData = new FormData(contactForm);
      const formObject = Object.fromEntries(formData);
      console.log('Form submitted:', formObject);
      alert('Thank you for your message! I\'ll get back to you soon.');
      contactForm.reset();
  });

  // Add fade-in animation to sections
  const sections = document.querySelectorAll('section');
  const fadeInOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -100px 0px"
  };

  const fadeInObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
          if (entry.isIntersecting) {
              entry.target.style.opacity = 1;
              entry.target.style.transform = 'translateY(0)';
              observer.unobserve(entry.target);
          }
      });
  }, fadeInOptions);

  sections.forEach(section => {
      section.style.opacity = 0;
      section.style.transform = 'translateY(20px)';
      section.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
      fadeInObserver.observe(section);
  });
});

