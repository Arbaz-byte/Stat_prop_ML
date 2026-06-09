import numpy as np 
import matplotlib.pyplot as plt

# ====================================
# MODULE 1 : LOSS FUNCTION & GRADIENT
# ====================================

def loss_function(w1,w2):
    
    return (w1**2 + 2*w2**2)
    
def gardient(w1,w2):
    
    #Rate of chnage of loss when w1 is changing
    dw1=2*w1
    #Rate of change of loss when w2 is changing
    dw2=4*w2
    
    return dw1,dw2
        
        
# ==========================================
# MODULE 2 GRADIENT DECENT 2D
# ==========================================

loss_hist=[]
def GD_2D(w1_init,w2_init,learning_rate,n_iter):
    
    #Initial weights
    w1,w2 = w1_init,w2_init
    
    ax.scatter(w1,w2,color='red',edgecolors='black',zorder=5)
    ax.text(w1, w2, f'  ({w1:.2f}, {w2:.2f})', fontsize=8, ha='left', va='bottom')
    
    
    for step in range(1,n_iter+1):
        
        w1_old,w2_old = w1,w2 
        
        #calculating the Gradient of w1 and w2
        dw1,dw2=gardient(w1,w2)
        
        #Updating w1 and w2
        w1-=learning_rate*dw1
        w2-=learning_rate*dw2
        
        ax.plot([w1_old,w1],[w2_old,w2],linewidth=2)
        ax.arrow(w1_old, w2_old, w1-w1_old, w2-w2_old, head_width=0.08, head_length=0.06, fc='white', ec='orange', zorder=4)
        
        current_loss=loss_function(w1,w2)
        ax.set_title(f"Iteration {step} | Loss {current_loss:.4f}")
        plt.pause(0.5)
    loss_hist.append(current_loss)    
    
    # Mark the last point with a star
    ax.scatter(w1, w2, marker='*', color='yellow', edgecolors='black', s=50, zorder=6)
        
    return w1,w2

#Generating Grid for Contour Plot
w1_vals = np.linspace(-2.5, 2.5, 100)
w2_vals = np.linspace(-2.5,2.5, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = loss_function(W1, W2)


#Creating the Visualization
fig,ax = plt.subplots(figsize=(9, 7))

# Ploting contours with a smooth filled color map
contours = ax.contourf(W1, W2, Z, levels=25, cmap='viridis', alpha=0.85)
plt.colorbar(contours, label='Loss Value $L(w_1, w_2)$')

# Adding specific contour lines for structural clarity
line_contours = ax.contour(W1, W2, Z, levels=20, colors='white', alpha=0.3, linewidths=1)
    
ax.clabel(line_contours, inline=True, fontsize=8)                           


ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,2.5)
ax.set_xlabel('$W_1$')
ax.set_ylabel('$W_2$')

# Now call the function multiple times on the same plot
lr = 0.1
iteration = 20
w1 = [1.5, -1.5,2.0]
w2 = [1.5, 1.5,-1.0]

colors = ['orange', 'cyan']  # Different colors for each trajectory

for i in range(len(w1)):
    w1_final, w2_final = GD_2D(w1[i], w2[i], lr, iteration)

ax.set_title(f"Loss {loss_hist[0]:.4f} and {loss_hist[1]:.4f}")
plt.show()

    
 
#Generating Grid for Contour Plot
w1_vals = np.linspace(-2.5, 2.5, 100)
w2_vals = np.linspace(-2.5,2.5, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = loss_function(W1, W2)


#Creating the Visualization
fig,ax = plt.subplots(figsize=(9, 7))

# Ploting contours with a smooth filled color map
contours = ax.contourf(W1, W2, Z, levels=25, cmap='viridis', alpha=0.85)
plt.colorbar(contours, label='Loss Value $L(w_1, w_2)$')

# Adding specific contour lines for structural clarity
line_contours = ax.contour(W1, W2, Z, levels=20, colors='white', alpha=0.3, linewidths=1)
    
ax.clabel(line_contours, inline=True, fontsize=8)                           


ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,2.5)
ax.set_xlabel('$W_1$')
ax.set_ylabel('$W_2$')    