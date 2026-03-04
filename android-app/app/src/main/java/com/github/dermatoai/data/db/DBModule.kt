package com.github.dermatoai.data.db

import android.content.Context
import androidx.room.Room
import com.github.dermatoai.data.db.dao.PredictionRecordDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DBModule {

    private const val DATABASE_NAME = "dermato_ai.db"

    @Provides
    @Singleton
    fun provideDatabase(
        @ApplicationContext context: Context
    ): AppDatabase =
        Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            DATABASE_NAME
        )
            .build()

    @Provides
    fun providePredictionDao(db: AppDatabase): PredictionRecordDao =
        db.predictionDao()
}
