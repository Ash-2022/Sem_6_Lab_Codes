// institutes_app/static/institutes_app/js/script.js
$(document).ready(function() {
    // Array to store institute data
    const institutes = [
        {% for institute in institutes %}
            {
                id: {{ institute.institute_id }},
                name: "{{ institute.name }}",
                courses: {{ institute.no_of_courses }}
            },
        {% endfor %}
    ];
    
    // Handle select change
    $('#instituteSelect').on('change', function() {
        const selectedId = $(this).val();
        const selectedInstitute = institutes.find(inst => inst.id == selectedId);
        
        if (selectedInstitute) {
            $('#instituteName').text(selectedInstitute.name);
            $('#coursesCount').text(selectedInstitute.courses);
            $('#instituteInfo').removeClass('d-none').hide().fadeIn(300);
        }
    });
    
    // Add hover effect
    $('#instituteSelect option').hover(
        function() {
            $(this).addClass('bg-light');
        },
        function() {
            $(this).removeClass('bg-light');
        }
    );
    
    // Add animation when page loads
    $('.card').hide().fadeIn(500);
});
