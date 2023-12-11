import torch.nn as nn
import torch
import time
import os

def train(model, train_loader, valid_loader, learning_rate, max_epoch, saved_epoch, device):

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Move the model to device
    model = nn.DataParallel(model)
    model = model.to(device)

    # Set up variables to recode losses and accuracies later
    train_loss_overtime = []
    val_loss   = []
    val_acc    = []

    # TODO: set up the optimizer and the loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # TODO: implement the training procedure. Please remember to zero out previous gradients in each iteration.
    
    # TODO: within the training loop, please calculate and record the validation loss and accuracy
    print("Training Start:")
    for epoch in range(max_epoch):
        output_epoch = epoch + saved_epoch
        model.train() # training mode
        total_loss = 0.0
        start = time.time()
        # all training data
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            #zero out previous gradients in each iteration.s
            optimizer.zero_grad()

            #forward
            outputs = model(images)
            # calculate the loss
            loss = loss_fn(outputs, labels.long())

            loss.backward() #backpropagation

            optimizer.step() #update parameters in model
            
            total_loss += loss.item() #convert to scaler

            #every 45 print
            # if batch_i % 45 == 44:    # print every 45 batches
            # avg_loss = running_loss/45
            # # record and print the avg loss over the 100 batches
            # loss_over_time.append(avg_loss)
            # print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(output_epoch + 1, batch_i+1, avg_loss))
            # running_loss = 0.0

        # avg loss for each epoch
        time_s = (', time spent: %.2f sec' %(time.time()-start))
        avg_train_loss = total_loss / len(train_loader) #batch number
        train_loss_overtime.append(avg_train_loss)
        #print(f"Epoch [{output_epoch+1}/{max_epoch}]: Training Loss: {avg_train_loss:.4f}, Validation Loss: NOT ready, Accuracy: NOT ready" + time_s)
        
        #on the val set ############
        model.eval()
        total = 0
        correct = 0
        total_val_loss = 0.0

        #calculate and record the validation loss and accuracy
        with torch.no_grad():
            for images, labels in valid_loader: #per batch
                images, labels = images.to(device), labels.to(device)
        
                outputs = model(images)
                loss = loss_fn(outputs, labels.long())
                #print(outputs.data)
                maxScore, pred_idx = torch.max(outputs.data, 1) #category of max score 

                total += labels.size(0)  #how many images per batch, sum them up for acc
                correct += (pred_idx == labels).sum().item()  
                #if won't work 1. torser 2. might be many elements correct in it
                #use .sum() return numbers of true (tonsor) .ietm() get scaler
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        val_loss.append(avg_val_loss)

        accuracy = (correct / total) * 100
        val_acc.append(accuracy)
        #If you want, you can uncomment this line to print losses
        print(f"Epoch [{output_epoch+1}/{max_epoch}]: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%" + time_s)


        saved_models_dir = 'saved_models'
        if not os.path.exists(saved_models_dir):
            os.makedirs(saved_models_dir)
        if output_epoch % 100 == 99: # save every 100 epochs
            torch.save(model.state_dict(), 'saved_models/GoogLeNet_{}.pt'.format(output_epoch + 1))
            print(f"model {'saved_models/GoogLeNet_{}.pt'.format(output_epoch + 1)} saved..")
    #return train_loss_overtime, val_loss, val_acc
    return train_loss_overtime, val_loss, val_acc