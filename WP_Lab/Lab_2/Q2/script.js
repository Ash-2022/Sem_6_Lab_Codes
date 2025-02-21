// script.js
$(document).ready(function() {
    const ball = $('.ball');
    const container = $('.container');
    const containerHeight = container.height();
    const ballHeight = ball.height();

    let posY = 0; // Start at the top
    let velocityY = 0; // Initial vertical velocity
    const gravity = 0.5; // Gravity effect
    const bounceFactor = -0.7; // Bounce reduction factor
    const velocityThreshold = 1; // Threshold to determine when the ball has "stopped"

    function animateBall() {
        velocityY += gravity;
        posY += velocityY;

        if (posY + ballHeight > containerHeight) {
            posY = containerHeight - ballHeight; 
            velocityY *= bounceFactor; 
        }

        if (posY < 0) {
            posY = 0; 
            velocityY *= bounceFactor; 
        }

        ball.css({
            top: posY
        });

        if (Math.abs(velocityY) < velocityThreshold && posY + ballHeight >= containerHeight) {
            resetBall(); 
        }

        requestAnimationFrame(animateBall);
    }

    function resetBall() {
        posY = 0; 
        velocityY = 5; 
    }

    resetBall(); 
    animateBall();
});
