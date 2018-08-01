import  tensorflow as tf

def heatmap_loss(g_out,g_gt, r_out,r_gt,
                 global_weights,refine_weights,
                 global_ohkm_n, refine_ohkm_n,
                 batch_size,name='heatmap_loss'):
        with tf.name_scope('global_loss'):
            # loss_heatmap = tf.nn.sigmoid_cross_entropy_with_logits(logits=g_out,
            #                                                        labels=g_gt)
            loss_heatmap = tf.square(g_out-g_gt)
            w1 = tf.expand_dims(global_weights, axis=1)
            w2 = tf.expand_dims(w1, axis=1)
            w3 = tf.expand_dims(w2, axis=1)
            loss = tf.reduce_mean(tf.multiply(w3, loss_heatmap), axis=[0, 1, 2, 3])
            ohkm_loss, _ = tf.nn.top_k(loss, global_ohkm_n)
            #ohkm_loss, _ = tf.nn.top_k(ohkm_loss, global_ohkm_n - 1) ## drop the lagest one
            ohkm_loss = tf.reduce_mean(ohkm_loss)
            ohem_n = int(batch_size * 0.8)
            loss = tf.reduce_mean(tf.multiply(w3, loss_heatmap), axis=[1, 2, 3, 4])
            ohem_loss, _ = tf.nn.top_k(loss, ohem_n)
            #ohem_loss, _ = tf.nn.top_k(ohem_loss, ohem_n - 1)## drop the lagest one
            ohem_loss = tf.reduce_mean(ohem_loss)
            global_loss = tf.add(ohem_loss, ohkm_loss, name='global_loss')
            tf.summary.scalar('global_ohkm_loss', ohkm_loss)
            tf.summary.scalar('global_ohem_loss', ohem_loss)
            tf.summary.scalar('global_total_loss', global_loss)

        with tf.name_scope('refine_loss'):
            # loss_heatmap = tf.nn.sigmoid_cross_entropy_with_logits(logits=r_out,
            #                                                        labels=r_gt)
            loss_heatmap= tf.square(r_out - r_gt)
            w1 = tf.expand_dims(refine_weights, axis=1)
            w2 = tf.expand_dims(w1, axis=1)

            loss = tf.reduce_mean(tf.multiply(w2, loss_heatmap), axis=[0, 1, 2])
            ohkm_loss, _ = tf.nn.top_k(loss, refine_ohkm_n)
            #ohkm_loss, _ = tf.nn.top_k(ohkm_loss, refine_ohkm_n - 1)  ## drop the lagest one
            ohkm_loss = tf.reduce_mean(ohkm_loss)


            ohem_n = int(batch_size * 0.5)
            loss = tf.reduce_mean(tf.multiply(w2, loss_heatmap), axis=[1, 2, 3])
            ohem_loss, _ = tf.nn.top_k(loss, ohem_n)
            #ohem_loss, _ = tf.nn.top_k(ohem_loss, ohem_n - 1)  ## drop the lagest one
            ohem_loss = tf.reduce_mean(ohem_loss)
            tf.summary.scalar('refine_ohkm_loss', ohkm_loss)
            tf.summary.scalar('refine_ohem_loss', ohem_loss)
            refine_loss =8*tf.add(ohem_loss, ohkm_loss, name='refine_loss')
            tf.summary.scalar('refine_total_loss', refine_loss)

        heatmap_total_loss = global_loss + refine_loss
        tf.summary.scalar('heatmap_total_loss', heatmap_total_loss)
        return heatmap_total_loss



